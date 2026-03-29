# QLoRA Fine-Tuning
I used [Quantized LoRA (QLoRA)] to adapt facebook/opt-125m on a ~2,280-sample subset of Databricks Dolly-15k (instruction-response format).

## Key achievements:
- Loaded the model in 4-bit NF4 quantization + double quantization
- Applied LoRA only on q_proj & v_proj → trainable parameters dropped to ~0.18% 
- Trained for 10 epochs on free Kaggle P100 GPU with gradient checkpointing & paged optimizer
- Reached validation perplexity ≈ 13
- Built an interactive Gradio demo with temperature & max-token sliders

## Problems I faced & solved along the way:
1. Loss computed on the full prompt → caused heavy template repetition & looping. Fixed with proper label masking (only response tokens contribute to loss).
2. Generation repetition & filler text → "ah ok thanks" everywhere. Solved with stronger repetition_penalty (1.3), no_repeat_ngram_size=4, lower temperature & top-p.
3. Gradio UI issues(missing buttons, format errors with ChatInterface, deprecation warnings). 
Ended up switching back to classic gr.Interface for the familiar sliders + big response box.

The generations are still limited (tiny model + small data = expected), but the real win was mastering the full PEFT + quantization + deployment workflow end-to-end.
