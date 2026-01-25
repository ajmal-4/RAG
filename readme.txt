1. git clone https://github.com/ggerganov/llama.cpp
2. cd llama.cpp

3. cmake -B build
4. cmake --build build --config Release

5. python convert_hf_to_gguf.py C:\Users\ajmal\Downloads\results\qwen_tool_router_merged --outfile qwen-tool-router-f16.gguf --outtype f16
6. build\bin\Release\llama-quantize.exe qwen-tool-router-f16.gguf qwen-tool-router-q4.gguf q4_k_m

Base Model
---------

Qwen/Qwen2.5-3B-Instruct

Merged Fine-Tuned Model
-----------------------

qwen-tool-router-merged-fp16

Quantized Model
---------------

GGUF q4 Quantized Model

qwen-tool-router-q4.gguf

Overall
---------
Fine-tuned using LoRA (PEFT)

Merged into standalone FP16 model

Converted and quantized to GGUF for production deployment