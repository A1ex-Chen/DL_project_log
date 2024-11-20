@nox.session(python=['3.8', '3.9', '3.10', '3.11'])
def tests(session):
    session.install('.[cli,api,assets-gcs,assets-s3,assets-az]')
    session.install('-r', 'requirements-dev.txt')
    session.run('pytest', '--junitxml=junit.xml', '-s')
