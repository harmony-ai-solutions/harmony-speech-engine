# OpenAPI Client Generator Script for Harmony Speech Engine
# Language: Golang
#
# Generation options for Golang can be found here:
# https://github.com/OpenAPITools/openapi-generator/blob/master/docs/generators/go.md
# https://stackoverflow.com/questions/75405511/the-line-import-openapiclient-github-com-git-user-id-git-repo-id-is-added-when

mkdir -p goclient

docker run --rm \
  -v ${PWD}:/local openapitools/openapi-generator-cli generate \
  -i /local/openapi.json \
  -g go \
  -p disallowAdditionalPropertiesIfNotPresent=false \
  -p packageName=harmonyspeech \
  -p packageVersion=v0.1.0 \
  -o /local/goclient \
  --git-repo-id harmony-speech-engine-client-go \
  --git-user-id harmony-ai-solutions
