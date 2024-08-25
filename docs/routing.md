# Request Routing Mechanism

Harmony Speech Engine aims at providing a hollistic approach at hosting different Neural Network models
which provide functionality for AI Speech related tasks.

A lot of current AI Speech Frameworks consist of multiple models or weight elements, which perform different
tasks or steps in the speech processing of the framework. Harmony Speech Engine allows for hosting and scaling the
individual models independently, and requests to a specific framework will be routed between the individual models
as required.

The following diagram illustrates the Harmony Speech Engine Routing Mechanism for two currently implemented Speech
Frameworks, "Harmony Speech" and "OpenVoice":

=> TODO: Add Diagram here

Also, tasks like embedding a speaker or transcribing an audio file are not always required as part of the processing,
and calling them repeatedly causes unnecessary compute time and might slow down an inference task. Therefore,
Harmony Speech Engine provides funtionality in the routing to omit redundant steps, as well as the option of
calling a specific model directly, e.g. if you already have a speaker embedding or transcription metadata.