# quantization/ —— 模型量化（报告 6.2）

对应报告 6.2「模型量化」。

上传内容：
- INT8（W8A8）离线量化流程（LLM Compressor）
- GPTQ W4A16 离线量化流程（LLM Compressor）
- 运行时量化：bitsandbytes int8 / nf4 推理集成
- 量化对比与实测结论脚本（量化对比子集 770 条、运行时量化配对子集 590 条）

转换工具：LLM Compressor、bitsandbytes（参见报告附录「模型与版本」）。