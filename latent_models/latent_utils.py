import re
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, AutoModelForCausalLM, MBartTokenizerFast, MT5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
import CONSTANTS as CONSTANTS

from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent, MT5ForConditionalGenerationLatent



def get_latent_model(args):
    if 'bart' in args.enc_dec_model:
        config = BartForConditionalGeneration.from_pretrained(
            args.enc_dec_model).config
        lm = BARTForConditionalGenerationLatent.from_pretrained(
            args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            args.enc_dec_model)
    elif 't5' in args.enc_dec_model:
        if 'mt5' in args.enc_dec_model:
            config = MT5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            lm = MT5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
        else:
            config = T5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            lm = T5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
    else:
        print("Unsupported model")
        raise NotImplementedError

    # Freeze all parameters initially
    for name, param in lm.named_parameters():
        param.requires_grad = False

    if getattr(args, 'use_lora', False):
        # Only train LoRA parameters
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        lm = get_peft_model(lm, lora_config)
        for name, param in lm.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                print(f"[LoRA] Trainable: {name}")
        print("LoRA has been applied to the encoder-decoder model.")
    elif args.lm_mode == 'ft':
        # Full fine-tuning
        for name, param in lm.named_parameters():
            param.requires_grad = True
    elif args.lm_mode == 'freeze':
        # Only train perceiver module
        for name, param in lm.named_parameters():
            if re.fullmatch(".*perceiver.*", name):
                param.requires_grad = True
                print(f"[Freeze Mode] Trainable: {name}")
    else:
        raise NotImplementedError

    return lm, tokenizer, config