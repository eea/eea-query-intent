pipeline {
  agent any

  environment {
    IMAGE_NAME = "eeacms/eea-query-intent"
    GIT_NAME = "eea-query-intent"
  }

  stages {

    stage('Build, Test & Push') {
      steps {
        node(label: 'docker-big-jobs') {
          script {
            checkout scm
            def isPr = env.CHANGE_ID && env.CHANGE_ID != ''
            def shortSha = env.GIT_COMMIT.take(7)
            if (isPr) {
              // PR build (BRANCH_NAME is the target branch): test only, never pushed
              tagName = "pr-${env.CHANGE_ID}"
            } else if (env.BRANCH_NAME == 'master') {
              // default branch (renamed from main): push latest + commit sha
              tagName = 'latest'
            } else {
              // git tag build (BRANCH_NAME is the tag name) or feature branch
              tagName = env.BRANCH_NAME
            }
            try {
              def image = docker.build("${IMAGE_NAME}:${tagName}", "--no-cache .")
              // run the 27-test suite inside the built image (it carries all
              // runtime deps; the tests use a fake adapter, no model needed)
              sh "docker run --rm -v \$(pwd):/repo -w /repo ${IMAGE_NAME}:${tagName} sh -c 'pip install -q pytest && python -m pytest'"
              if (!isPr && (env.BRANCH_NAME == 'master' || buildingTag())) {
                docker.withRegistry('', 'eeajenkins') {
                  image.push()
                  if (env.BRANCH_NAME == 'master') {
                    image.push(shortSha)
                  }
                }
              }
            } finally {
              sh "docker rmi ${IMAGE_NAME}:${tagName}"
            }
          }
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
