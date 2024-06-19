# الاستدلال باستخدام وحدة معالجة الرسوميات (GPU)

تعد وحدات معالجة الرسوميات (GPU) الخيار القياسي لأجهزة تعلم الآلة، على عكس وحدات المعالجة المركزية (CPU)، لأنها مُحسّنة لعرض نطاق الذاكرة والتوازي. ولمواكبة الأحجام الأكبر للنماذج الحديثة أو لتشغيل هذه النماذج الكبيرة على الأجهزة الموجودة والقديمة، هناك العديد من التحسينات التي يمكنك استخدامها لتسريع الاستدلال باستخدام وحدة معالجة الرسوميات. في هذا الدليل، ستتعلم كيفية استخدام FlashAttention-2 (آلية اهتمام أكثر كفاءة في استخدام الذاكرة)، وBetterTransformer (مسار تنفيذ سريع أصلي في PyTorch)، وbitsandbytes لضغط نموذجك إلى دقة أقل. وأخيرًا، تعرف على كيفية استخدام 🤗 Optimum لتسريع الاستدلال باستخدام ONNX Runtime على وحدات معالجة الرسوميات Nvidia وAMD.

<Tip>

تنطبق معظم التحسينات الموضحة هنا أيضًا على إعدادات متعددة لوحدات معالجة الرسوميات!

</Tip>

## FlashAttention-2

<Tip>

FlashAttention-2 تجريبي وقد يتغير بشكل كبير في الإصدارات المستقبلية.

</Tip>

[FlashAttention-2](https://huggingface.co/papers/2205.14135) هو تنفيذ أسرع وأكثر كفاءة لآلية الاهتمام القياسية التي يمكن أن تسرع الاستدلال بشكل كبير من خلال:

1. موازاة حساب الاهتمام بشكل إضافي على طول التسلسل.
2. تقسيم العمل بين خيوط وحدة معالجة الرسوميات لتقليل التواصل وقراءات/كتابات الذاكرة المشتركة بينها.

تدعم FlashAttention-2 حاليًا المعماريات التالية:

- [Bark](https://huggingface.co/docs/transformers/model_doc/bark#transformers.BarkModel)
- [Bart](https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartModel)
- [Cohere](https://huggingface.co/docs/transformers/model_doc/cohere#transformers.CohereModel)
- [Dbrx](https://huggingface.co/docs/transformers/model_doc/dbrx#transformers.DbrxModel)
- [DistilBert](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel)
- [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.GemmaModel)
- [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#transformers.GPTBigCodeModel)
- [GPTNeo](https://huggingface.co/docs/transformers/model_doc/gpt_neo#transformers.GPTNeoModel)
- [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXModel)
- [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj#transformers.GPTJModel)
- [Idefics2](https://huggingface.co/docs/transformers/model_doc/idefics2#transformers.Idefics2Model)
- [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconModel)
- [JetMoe](https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeModel)
- [Jamba](https://huggingface.co/%D8%A7%D9%84%D9%85%D9%84فات/model_doc/jamba#transformers.JambaModel)
- [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)
- [Llava](https://huggingface.co/docs/transformers/model_doc/llava)
- [Llava-NeXT](https://huggingface.co/docs/transformers/model_doc/llava_next)
- [VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)
- [VideoLlava](https://huggingface.co/docs/transformers/model_doc/video_llava)
- [M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)
- [MBart](https://huggingface.co/docs/transformers/model_doc/mbart#transformers.MBartModel)
- [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
- [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
- [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
- [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
- [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)
- [OLMo](https://huggingface.co/docs/transformers/model_doc/olmo#transformers.OlmoModel)
- [OPT](https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTModel)
- [Phi](https://huggingface.co/docs/transformers/model_doc/phi#transformers.PhiModel)
- [Phi3](https://huggingface.co/docs/transformers/model_doc/phi3#transformers.Phi3Model)
- [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
- [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
- [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model)
- [Qwen2MoE](https://huggingface.co/docs/transformers/model_doc/qwen2_moe#transformers.Qwen2MoeModel)
- [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
- [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
- [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel)
- [data2vec_audio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecAudioModel)
- [Sew](https://huggingface.co/docs/transformers/main/en/model_doc/sew#transformers.SEWModel)
- [UniSpeech](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech#transformers.UniSpeechModel)
- [unispeech_sat](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel)

يمكنك طلب إضافة دعم FlashAttention-2 لنموذج آخر من خلال فتح مشكلة أو طلب سحب على GitHub.

قبل البدء، تأكد من تثبيت FlashAttention-2.

لتمكين FlashAttention-2، مرر الحجة `attn_implementation="flash_attention_2"` إلى [`~AutoModelForCausalLM.from_pretrained`]:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
model_id,
torch_dtype=torch.bfloat16,
attn_implementation="flash_attention_2",
)
```

<Tip>

يمكن استخدام FlashAttention-2 فقط عندما يكون نوع بيانات النموذج `fp16` أو `bf16`. تأكد من تحويل نموذجك إلى نوع البيانات المناسب وتحميله على جهاز مدعوم قبل استخدام FlashAttention-2.

<br>

يمكنك أيضًا تعيين `use_flash_attention_2=True` لتمكين FlashAttention-2 ولكنها أصبحت مهملة لصالح `attn_implementation="flash_attention_2"`.

</Tip>

يمكن الجمع بين FlashAttention-2 وتقنيات التحسين الأخرى مثل الضغط لزيادة تسريع الاستدلال. على سبيل المثال، يمكنك الجمع بين FlashAttention-2 والضغط 8-بت أو 4-بت:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load in 8bit
model = AutoModelForCausalLM.from_pretrained(
model_id,
load_in_8bit=True,
attn_implementation="flash_attention_2",
)

# load in 4bit
model = AutoModelForCausingLM.from_pretrained(
model_id,
load_in_4bit=True,
attn_implementation="flash_attention_2",
)
```
### تسريع الأداء المتوقع

يمكنك الاستفادة من تسريع الأداء بشكل كبير عند الاستنتاج، خاصة بالنسبة للمدخلات ذات التسلسلات الطويلة. ومع ذلك، نظرًا لأن FlashAttention-2 لا يدعم حساب درجات الاهتمام مع رموز الحشو، يجب عليك يدويًا حشو/إلغاء حشو درجات الاهتمام للاستنتاج المجمع عندما تحتوي التسلسل على رموز حشو. يؤدي هذا إلى تباطؤ كبير في الأجيال المجمعة مع رموز الحشو.

لتجاوز ذلك، يجب استخدام FlashAttention-2 بدون رموز حشو في التسلسل أثناء التدريب (عن طريق حزم مجموعة بيانات أو [ربط التسلسلات](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516) حتى الوصول إلى طول التسلسل الأقصى).

بالنسبة لمرور أمامي واحد على [tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b) بطول تسلسل يبلغ 4096 وأحجام دفعات مختلفة بدون رموز حشو، يكون تسريع الأداء المتوقع على النحو التالي:

بالنسبة لمرور أمامي واحد على [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) بطول تسلسل يبلغ 4096 وأحجام دفعات مختلفة بدون رموز حشو، يكون تسريع الأداء المتوقع على النحو التالي:

بالنسبة للتسلسلات التي تحتوي على رموز حشو (توليد باستخدام رموز حشو)، يجب إلغاء حشو/حشو تسلسلات المدخلات لحساب درجات الاهتمام بشكل صحيح. باستخدام طول تسلسل صغير نسبيًا، يؤدي المرور الأمامي الواحد إلى زيادة العبء تؤدي إلى تسريع الأداء الطفيف (في المثال أدناه، يتم ملء 30% من الإدخال برموز الحشو):

ولكن بالنسبة لأطوال التسلسلات الأكبر، يمكنك توقع فوائد أكبر من تسريع الأداء:

> يعد FlashAttention أكثر كفاءة في استخدام الذاكرة، مما يعني أنه يمكنك التدريب على أطوال تسلسلات أكبر بكثير دون مواجهة مشكلات نفاد الذاكرة. يمكنك تقليل استخدام الذاكرة بنسبة تصل إلى 20 ضعفًا لأطوال التسلسلات الأكبر. اطلع على مستودع [flash-attention](https://github.com/Dao-AILab/flash-attention) لمزيد من التفاصيل.

يدعم PyTorch [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) أيضًا استدعاء FlashAttention ونواة الاهتمام الموفرة للذاكرة في الخلفية. يجري حاليًا إضافة دعم SDPA بشكل أصلي في Transformers ويتم استخدامه بشكل افتراضي لـ `torch>=2.1.1` عند توفر التنفيذ. يمكنك أيضًا تعيين `attn_implementation="sdpa"` في `from_pretrained()` لطلب استخدام SDPA بشكل صريح.

في الوقت الحالي، يدعم Transformers الاستدلال والتدريب SDPA للعمارات التالية:

- [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTModel)
- [Bart](https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartModel)
- [Bert](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel)
- [Cohere](https://huggingface.co/docs/transformers/model_doc/cohere#transformers.CohereModel)
- [Dbrx](https://huggingface.co/docs/transformers/model_doc/dbrx#transformers.DbrxModel)
- [DeiT](https://huggingface.co/docs/transformers/model_doc/deit#transformers.DeiTModel)
- [Dpr](https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DprReader)
- [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconModel)
- [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.GemmaModel)
- [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#transformers.GPTBigCodeModel)
- [JetMoe](https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeModel)
- [Jamba](https://huggingface.co/docs/transformers/model_doc/jamba#transformers.JambaModel)
- [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)
- [OLMo](https://huggingface.co/docs/transformers/model_doc/olmo#transformers.OlmoModel)
- [PaliGemma](https://huggingface.co/docs/transformers/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration)
- [Phi](https://huggingface.co/docs/transformers/model_doc/phi#transformers.PhiModel)
- [Idefics](https://huggingface.co/docs/transformers/model_doc/idefics#transformers.IdeficsModel)
- [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
- [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
- [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
- [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
- [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
- [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model)
- [Qwen2MoE](https://huggingface.co/docs/transformers/model_doc/qwen2_moe#transformers.Qwen2MoeModel)
- [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
- [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
- [ViT](https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTModel)
- [ViTHybrid](https://huggingface.co/docs/transformers/model_doc/vit_hybrid#transformers.ViTHybridModel)
- [ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEModel)
- [ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn#transformers.ViTMSNModel)
- [VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEModell)
- [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
- [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel)
- [data2vec_audio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecAudioModel)
- [Sew](https://huggingface.co/docs/transformers/main/en/model_doc/sew#transformers.SEWModel)
- [UniSpeech](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech#transformers.UniSpeechModel)
- [unispeech_sat](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel)
- [YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos#transformers.YolosModel)

> يمكن استخدام FlashAttention فقط للنماذج ذات النوع `fp16` أو `bf16` torch، لذا تأكد من تحويل نموذجك إلى النوع المناسب أولاً. يمكن لمنصة الاهتمام الموفرة للذاكرة التعامل مع نماذج `fp32`.

> لا يدعم SDPA مجموعات معينة من معلمات الاهتمام، مثل `head_mask` و`output_attentions=True`. في هذه الحالة، يجب أن تشاهد رسالة تحذير وسنعود إلى التنفيذ (الأبطأ).

بشكل افتراضي، يختار SDPA نواة الأداء الأكثر كفاءة المتاحة، ولكن يمكنك التحقق مما إذا كانت منصة متوفرة في إعداد معين (الأجهزة، وحجم المشكلة) باستخدام [`torch.backends.cuda.sdp_kernel`](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) كمدير سياق:

إذا رأيت خطأ مع تتبع المكدس أدناه، فجرّب استخدام الإصدار الليلي من PyTorch الذي قد يكون له تغطية أوسع لـ FlashAttention:

```bash
RuntimeError: No available kernel. Aborting execution.

# install PyTorch nightly
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```
## BetterTransformer

يوفر BetterTransformer تسريعًا للاستدلال من خلال تنفيذ fastpath (تنفيذ متخصص لـ PyTorch الأصلي لوظائف Transformer). هناك تحسينان في تنفيذ fastpath:

1. الاندماج، الذي يجمع بين عدة عمليات متتالية في "kernel" واحد لتقليل عدد خطوات الحساب.
2. تخطي ندرة التوكينز الفارغة المتأصلة لتجنب الحسابات غير الضرورية مع التنسورات المُعشَّشة.

يحول BetterTransformer أيضًا جميع عمليات الانتباه لاستخدام [scaled dot product attention (SDPA)](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) الأكثر كفاءة في الذاكرة، كما أنه يستدعي كيرنلات مُحسَّنة مثل [FlashAttention](https://huggingface.co/papers/2205.14135) في الخلفية.

قبل أن تبدأ، تأكد من تثبيت 🤗 Optimum [installed](https://huggingface.co/docs/optimum/installation).

بعد ذلك، يمكنك تمكين BetterTransformer باستخدام طريقة [`PreTrainedModel.to_bettertransformer`]:

```python
model = model.to_bettertransformer()
```

يمكنك إعادة نموذج Transformers الأصلي باستخدام طريقة [`~PreTrainedModel.reverse_bettertransformer`]. يجب استخدام هذا قبل حفظ نموذجك لاستخدام النمذجة القياسية لـ Transformers:

```py
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

## bitsandbytes

bitsandbytes هي مكتبة للتحويل الكمي تتضمن دعمًا للتحويل الكمي 4-بت و8-بت. يقلل التحويل الكمي حجم نموذجك مقارنة بإصداره الكامل الدقة الأصلي، مما يسهل وضع النماذج الكبيرة على وحدات معالجة الرسومات (GPU) ذات الذاكرة المحدودة.

تأكد من تثبيت bitsandbytes و🤗 Accelerate:

```bash
# هذه الإصدارات تدعم 8-بت و4-بت
pip install bitsandbytes>=0.39.0 accelerate>=0.20.0

# تثبيت Transformers
pip install transformers
```

### 4-بت

لتحميل نموذج في 4-بت للاستدلال، استخدم معلمة `load_in_4bit`. معلمة `device_map` اختيارية، ولكن يوصى بتعيينها إلى `"auto"` للسماح لـ 🤗 Accelerate بتخصيص النموذج تلقائيًا وبكفاءة بالنظر إلى الموارد المتاحة في البيئة.

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```

لتحميل نموذج في 4-بت للاستدلال باستخدام وحدات معالجة رسومات متعددة، يمكنك التحكم في مقدار ذاكرة GPU التي تريد تخصيصها لكل GPU. على سبيل المثال، لتوزيع 600 ميجابايت من الذاكرة على GPU الأول و1 جيجابايت من الذاكرة على GPU الثاني:

```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
model_name, device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
)
```

### 8-بت

إذا كنت فضوليًا ومهتمًا بمعرفة المزيد عن المفاهيم الأساسية للتحويل الكمي 8-بت، فاقرأ منشور المدونة [Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration).

لتحميل نموذج في 8-بت للاستدلال، استخدم معلمة `load_in_8bit`. معلمة `device_map` اختيارية، ولكن يوصى بتعيينها إلى `"auto"` للسماح لـ 🤗 Accelerate بتخصيص النموذج تلقائيًا وبكفاءة بالنظر إلى الموارد المتاحة في البيئة:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

إذا كنت تقوم بتحميل نموذج في 8-بت لتوليد النص، فيجب استخدام طريقة [`~transformers.GenerationMixin.generate`] بدلاً من وظيفة [`Pipeline`] التي لا تكون مُحسَّنة لنماذج 8-بت وستكون أبطأ. لا تدعم بعض استراتيجيات أخذ العينات، مثل أخذ العينات النووية، بواسطة [`Pipeline`] لنماذج 8-بت. يجب أيضًا وضع جميع المدخلات على نفس الجهاز مثل النموذج:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

لتحميل نموذج في 4-بت للاستدلال باستخدام وحدات معالجة رسومات متعددة، يمكنك التحكم في مقدار ذاكرة GPU التي تريد تخصيصها لكل GPU. على سبيل المثال، لتوزيع 1 جيجابايت من الذاكرة على GPU الأول و2 جيجابايت من الذاكرة على GPU الثاني:

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

جرب تشغيل نموذج T5 بحجم 11 مليار معلمة [T5 model] (https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing) أو نموذج BLOOM بحجم 3 مليارات معلمة [BLOOM model] (https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing) للاستدلال على وحدات معالجة الرسومات (GPU) من المستوى المجاني في Google Colab!

## 🤗 Optimum

لمعرفة المزيد من التفاصيل حول استخدام ORT مع 🤗 Optimum، راجع أدلة [Accelerated inference on NVIDIA GPUs](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#accelerated-inference-on-nvidia-gpus) و[Accelerated inference on AMD GPUs](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/amdgpu#accelerated-inference-on-amd-gpus). يقدم هذا القسم فقط مثالًا موجزًا وبسيطًا.

ONNX Runtime (ORT) هو مسرع نموذج يدعم الاستدلال المعجل على وحدات معالجة الرسومات (GPU) من Nvidia، ووحدات معالجة الرسومات (GPU) من AMD التي تستخدم [ROCm](https://www.amd.com/en/products/software/rocm.html) stack. يستخدم ORT تقنيات التحسين مثل دمج العمليات الشائعة في عقدة واحدة وطي الثوابت لتقليل عدد الحسابات التي يتم إجراؤها وتسريع الاستدلال. كما يضع ORT العمليات الأكثر كثافة حسابية على وحدة معالجة الرسومات (GPU) وبقية العمليات على وحدة المعالجة المركزية (CPU) لتوزيع عبء العمل بين الجهازين بذكاء.

يدعم 🤗 Optimum استخدام ONNX Runtime، والذي يمكن استخدامه في 🤗 Transformers. ستحتاج إلى استخدام [`~optimum.onnxruntime.ORTModel`] للمهمة التي تحاول حلها، وتحديد معلمة `provider` التي يمكن تعيينها إلى [`CUDAExecutionProvider`]، أو [`ROCMExecutionProvider`]، أو [`TensorrtExecutionProvider`]. إذا كنت تريد تحميل نموذج لم يتم تصديره بعد إلى ONNX، فيمكنك تعيين `export=True` لتحويل نموذجك أثناء التنقل إلى تنسيق ONNX:

```py
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
"distilbert/distilbert-base-uncased-finetuned-sst-2-english",
export=True,
provider="CUDAExecutionProvider",
)
```

الآن يمكنك استخدام النموذج للاستدلال:

```py
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

pipeline = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Both the music and visual were astounding, not to mention the actors performance.")
```

## الجمع بين التحسينات

غالبًا ما يكون من الممكن الجمع بين عدة تقنيات تحسين موصوفة أعلاه للحصول على أفضل أداء استدلالي ممكن لنموذجك. على سبيل المثال، يمكنك تحميل نموذج في 4-بت، ثم تمكين BetterTransformer مع FlashAttention:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# تحميل النموذج في 4-بت
quantization_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

# تمكين BetterTransformer
model = model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# تمكين FlashAttention
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```