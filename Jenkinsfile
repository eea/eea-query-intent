pipeline {
  agent {
    node { label 'docker-host' }
  }

  environment {
    // Hardcoded literal (not env.JOB_BASE_NAME — that resolves to the
    // branch name on multibranch jobs).
    GIT_NAME = "eea-query-intent"
    // Verified via `git remote show origin` — this repo's default branch
    // was renamed from main to master.
    DEFAULT_BRANCH = "master"
    IMAGE_NAME = BUILD_TAG.toLowerCase()
    TEST_IMAGE = "${IMAGE_NAME}-test"
    RELEASE_IMAGE = "${GIT_NAME}:${env.BUILD_NUMBER}"
    DOCKERHUB_IMAGE = "eeacms/eea-query-intent"
    TRIVY_IMAGE = "aquasec/trivy:latest"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Versioning') {
      steps {
        script {
          env.BASE_VERSION = sh(script: "grep -m1 '^version' pyproject.toml | cut -d'\"' -f2", returnStdout: true).trim()
          env.GIT_SHA_SHORT = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
          // On a tag build the image tag is always the git tag itself,
          // unconditionally — a mismatch with pyproject.toml is a warning,
          // not a failure (v-prefix, pre-releases, delayed bumps).
          if (env.TAG_NAME) {
            env.IMAGE_TAG = env.TAG_NAME
            def normalizedTag = env.TAG_NAME.replaceFirst(/^v/, '')
            if (env.BASE_VERSION != env.TAG_NAME && env.BASE_VERSION != normalizedTag) {
              echo "WARNING: git tag (${env.TAG_NAME}) does not match pyproject.toml version (${env.BASE_VERSION}) — pushing ${env.IMAGE_TAG} anyway. Bump pyproject.toml to match if this wasn't intentional."
              currentBuild.result = 'UNSTABLE'
            }
          } else {
            env.IMAGE_TAG = "${env.BASE_VERSION}-${env.BRANCH_NAME}-${env.BUILD_NUMBER}"
          }
        }
      }
    }

    stage('Build test image') {
      steps {
        sh '''docker build --pull -f Dockerfile.test -t $TEST_IMAGE .'''
      }
    }

    // No 'Auto-fix code style' stage — auto-fixing happens pre-commit with
    // these same commands; here they run read-only as a hard gate.
    stage('Code linting') {
      parallel {
        stage('Ruff check') {
          steps {
            sh '''docker run --rm $TEST_IMAGE uv run ruff check .'''
          }
        }
        stage('Ruff format') {
          steps {
            sh '''docker run --rm $TEST_IMAGE uv run ruff format --check .'''
          }
        }
      }
    }

    stage('Unit test') {
      steps {
        script {
          try {
            sh '''rm -rf xunit-reports-current && mkdir -p xunit-reports-current/coverage'''
            sh script: '''docker rm -f ${IMAGE_NAME}-unit''', returnStatus: true
            def status = sh(script: '''docker run --name="${IMAGE_NAME}-unit" $TEST_IMAGE uv run pytest --junitxml=junit.xml --cov=. --cov-report=lcov:coverage/lcov.info --cov-report=html:coverage/lcov-report --cov-report=xml:coverage/cobertura-coverage.xml''', returnStatus: true)
            sh '''docker cp ${IMAGE_NAME}-unit:/app/junit.xml xunit-reports-current/junit.xml'''
            sh '''docker cp ${IMAGE_NAME}-unit:/app/coverage/. xunit-reports-current/coverage'''
            // The Cobertura <sources> element carries the container CWD
            // (/app) as an absolute path; the SonarQube python sensor cannot
            // resolve it from the agent workspace ("Invalid directory path in
            // 'source' element" -> 0% coverage on older sensor versions). The
            // class filenames are already relative to the repo root, matching
            // sonar.sources=., so drop the element and let the sensor resolve
            // them against the project base directory.
            sh '''sed -i '/<sources>/,/<\\/sources>/d' xunit-reports-current/coverage/cobertura-coverage.xml'''
            publishHTML(target : [
              allowMissing: false,
              alwaysLinkToLastBuild: true,
              keepAll: true,
              reportDir: 'xunit-reports-current/coverage/lcov-report',
              reportFiles: 'index.html',
              reportName: 'UTCoverage',
              reportTitles: 'Unit Tests Code Coverage'
            ])
            if (status != 0) {
              error "unit tests failed"
            }
          } finally {
            // Plain junit step (no allowEmptyResults): the non-optional
            // docker cp above already guarantees a report exists; a missing
            // or malformed one must fail the stage, not be suppressed.
            junit testResults: 'xunit-reports-current/junit.xml'
            sh script: '''docker stop ${IMAGE_NAME}-unit''', returnStatus: true
            sh script: '''docker rm -v ${IMAGE_NAME}-unit''', returnStatus: true
          }
        }
      }
    }

    stage('Build release image') {
      steps {
        sh '''docker build --pull -t $RELEASE_IMAGE .'''
      }
    }

    stage('Integration test') {
      steps {
        script {
          try {
            sh '''rm -rf integration-reports-current && mkdir -p integration-reports-current'''
            sh script: '''docker rm -f ${IMAGE_NAME}-app''', returnStatus: true
            sh '''docker run -d --name="${IMAGE_NAME}-app" $RELEASE_IMAGE'''
            // ci_smoke.py waits for the model cold-start, then probes the
            // live service: health, one eligible query, one keyword query,
            // the empty/too-long/URL guards, and route/method rejection.
            def status = sh(script: '''
              docker cp scripts/ci_smoke.py ${IMAGE_NAME}-app:/tmp/ci_smoke.py
              docker exec ${IMAGE_NAME}-app python /tmp/ci_smoke.py
            ''', returnStatus: true)
            // ci_smoke.py always writes /tmp/junit-smoke.xml (including a
            // failed service_start testcase on cold-start timeout), so a
            // plain docker cp here fails the stage when the container died.
            sh '''docker cp ${IMAGE_NAME}-app:/tmp/junit-smoke.xml integration-reports-current/junit.xml'''
            junit testResults: 'integration-reports-current/junit.xml'
            if (status != 0) {
              error "integration smoke failed"
            }
          } finally {
            sh script: '''docker stop ${IMAGE_NAME}-app''', returnStatus: true
            sh script: '''docker rm -v ${IMAGE_NAME}-app''', returnStatus: true
          }
        }
      }
    }

    // Branches only (house pattern, like eea.genai.core): PR builds are not
    // analyzed, so no PR decoration is required on the SonarQube project.
    // (env.CHANGE_ID ?: '') also covers agents where the variable is absent
    // (null) rather than empty on non-PR builds.
    stage('Sonarqube test') {
      when {
        expression { (env.CHANGE_ID ?: '') == '' }
      }
      steps {
        script {
          def scannerHome = tool 'SonarQubeScanner'
          // Whole -D list precomputed here (house pattern) so the sh line
          // carries only explicit env.* interpolations.
          env.sonarParams = "-Dsonar.python.coverage.reportPaths=./xunit-reports-current/coverage/cobertura-coverage.xml -Dsonar.sources=. -Dsonar.projectKey=${env.GIT_NAME} -Dsonar.projectName=${env.GIT_NAME} -Dsonar.projectVersion=${env.BASE_VERSION} -Dsonar.branch.name=${env.BRANCH_NAME}"
          withSonarQubeEnv('Sonarqube') {
            // Python coverage goes to sonar.python.coverage.reportPaths as
            // Cobertura XML (never the JS LCOV property). sonar.sources is
            // the repo root so the fully-qualified coverage paths
            // (src/eea_query_intent/...) resolve to real files.
            sh "export PATH=${scannerHome}/bin:\$PATH; sonar-scanner ${env.sonarParams}"
          }
        }
      }
    }

    stage('Trivy test') {
      steps {
        // Full HIGH,CRITICAL report archived for visibility; only CRITICAL
        // fails the build (EEA policy — base-OS HIGHs are out of our
        // control). The report scan redirects to a file (then cats it) so
        // the scanner's own exit status is not masked by a pipe.
        sh '''
          mkdir -p trivy-reports
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            "$TRIVY_IMAGE" image --no-progress --format table --severity HIGH,CRITICAL \
            "$RELEASE_IMAGE" > trivy-reports/trivy-image.txt
          cat trivy-reports/trivy-image.txt
        '''
        archiveArtifacts artifacts: 'trivy-reports/*.txt', fingerprint: true, allowEmptyArchive: false
        sh '''
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            "$TRIVY_IMAGE" image --no-progress --severity CRITICAL --exit-code 1 \
            "$RELEASE_IMAGE"
        '''
      }
    }

    stage('Release on Docker Hub') {
      when {
        allOf {
          // (env.CHANGE_ID ?: '') also covers agents where the variable is
          // absent (null) rather than empty on non-PR builds.
          expression { (env.CHANGE_ID ?: '') == '' }
          anyOf {
            expression { env.BRANCH_NAME == env.DEFAULT_BRANCH }
            buildingTag()
          }
        }
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'jekinsdockerhub', usernameVariable: 'DOCKERHUB_USERNAME', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
          sh '''
            trap 'docker logout >/dev/null 2>&1 || true' EXIT
            echo "$DOCKERHUB_PASSWORD" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
            # A version-numbered tag is pushed only for a real git tag.
            if [ -n "${TAG_NAME:-}" ]; then
              docker tag "$RELEASE_IMAGE" "$DOCKERHUB_IMAGE:$IMAGE_TAG"
              docker push "$DOCKERHUB_IMAGE:$IMAGE_TAG"
            fi
            # :latest tracks the newest default-branch build; the short sha
            # tag lets Rancher pin an exact commit.
            if [ "$BRANCH_NAME" = "$DEFAULT_BRANCH" ]; then
              docker tag "$RELEASE_IMAGE" "$DOCKERHUB_IMAGE:latest"
              docker push "$DOCKERHUB_IMAGE:latest"
              docker tag "$RELEASE_IMAGE" "$DOCKERHUB_IMAGE:$GIT_SHA_SHORT"
              docker push "$DOCKERHUB_IMAGE:$GIT_SHA_SHORT"
            fi
            docker logout
          '''
        }
      }
    }
  }

  post {
    always {
      cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, deleteDirs: true)
    }
    changed {
      script {
        def details = """<h1>${env.JOB_NAME} - Build #${env.BUILD_NUMBER} - ${currentBuild.currentResult}</h1>
                         <p>Check console output at <a href="${env.BUILD_URL}/display/redirect">${env.JOB_BASE_NAME} - #${env.BUILD_NUMBER}</a></p>
                         """
        emailext(
        subject: '$DEFAULT_SUBJECT',
        body: details,
        attachLog: true,
        compressLog: true,
        recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'CulpritsRecipientProvider']]
        )
      }
    }
  }
}
