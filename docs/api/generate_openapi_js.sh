# OpenAPI Client Generator Script for Harmony Speech Engine
# Language: Golang
#
# Generation options for Golang can be found here:
# https://github.com/OpenAPITools/openapi-generator/blob/master/docs/generators/go.md
# https://stackoverflow.com/questions/75405511/the-line-import-openapiclient-github-com-git-user-id-git-repo-id-is-added-when

mkdir -p jsclient

docker run --rm \
  -v ${PWD}:/local openapitools/openapi-generator-cli generate \
  -i /local/openapi.json \
  -g javascript \
  -p disallowAdditionalPropertiesIfNotPresent=false \
  -p projectName=harmonyspeech \
  -p projectVersion=0.0.7 \
  -p usePromises=true \
  -o /local/jsclient \
  --git-repo-id harmony-speech-engine-client-js \
  --git-user-id harmony-ai-solutions
