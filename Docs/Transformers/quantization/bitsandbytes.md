# bitsandbytes

تعد [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) أسهل خيار لضغط نموذج إلى 8 بت و4 بت. تضاعف الضغط إلى 8 بت من تأثير القيم الشاذة على أداء النموذج من خلال ضرب القيم الشاذة في fp16 مع غير الشاذة في int8، وتحويل قيم غير الشاذة مرة أخرى إلى fp16، ثم جمعها معًا لإرجاع الأوزان في fp16. يضغط الضغط إلى 4 بت النموذج بشكل أكبر، ويشيع استخدامه مع [QLoRA](https://hf.co/papers/2305.14314) لضبط دقة النماذج اللغوية الكبيرة.

لاستخدام bitsandbytes، تأكد من تثبيت المكتبات التالية:

لضغط نموذج إلى 8 بت، تأكد من تثبيت المكتبات التالية:

```bash
pip install transformers accelerate bitsandbytes>0.37.0
```

لضغط نموذج إلى 4 بت، تأكد من تثبيت المكتبات التالية:

```bash
pip install bitsandbytes>=0.39.0
pip install --upgrade accelerate transformers
```

الآن يمكنك ضغط نموذج عن طريق تمرير `BitsAndBytesConfig` إلى طريقة [`~PreTrainedModel.from_pretrained`]. يعمل هذا مع أي نموذج في أي طريقة، طالما أنه يدعم التحميل باستخدام Accelerate ويحتوي على طبقات `torch.nn.Linear`.

لضغط نموذج إلى 8 بت، استخدم الكود التالي:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
"bigscience/bloom-1b7",
quantization_config=quantization_config
)
```

بشكل افتراضي، يتم تحويل جميع الوحدات الأخرى مثل `torch.nn.LayerNorm` إلى `torch.float16`. يمكنك تغيير نوع بيانات هذه الوحدات باستخدام معلمة `torch_dtype` إذا أردت:

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
"facebook/opt-350m",
quantization_config=quantization_config,
torch_dtype=torch.float32
)
model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

بمجرد ضغط نموذج إلى 8 بت، لا يمكنك دفع الأوزان المضغوطة إلى Hub إلا إذا كنت تستخدم أحدث إصدار من Transformers وbitsandbytes. إذا كان لديك أحدث الإصدارات، فيمكنك دفع النموذج المضغوط إلى 8 بت إلى Hub باستخدام طريقة [`~PreTrainedModel.push_to_hub`]. يتم أولاً دفع ملف التكوين، ثم تليها أوزان النموذج المضغوط.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
"bigscience/bloom-560m",
quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

لضغط نموذج إلى 4 بت، استخدم الكود التالي:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
"bigscience/bloom-1b7",
quantization_config=quantization_config
)
```

بشكل افتراضي، يتم تحويل جميع الوحدات الأخرى مثل `torch.nn.LayerNorm` إلى `torch.float16`. يمكنك تغيير نوع بيانات هذه الوحدات باستخدام معلمة `torch_dtype` إذا أردت:

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
"facebook/opt-350m",
quantization_config=quantization_config,
torch_dtype=torch.float32
)
model_4bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

إذا كان لديك إصدار `bitsandbytes>=0.41.3`، فيمكنك تسلسل النماذج المضغوطة إلى 4 بت ودفعها إلى Hugging Face Hub. ما عليك سوى استدعاء `model.push_to_hub()` بعد تحميله بدقة 4 بت. يمكنك أيضًا حفظ النماذج المضغوطة محليًا باستخدام أمر `model.save_pretrained()`.

<Tip warning={true}>  

تدعم التدريب باستخدام أوزان 8 بت و4 بت فقط لتدريب المعلمات *الإضافية*.  

</Tip>

يمكنك التحقق من البصمة الخاصة بك باستخدام طريقة `get_memory_footprint`:

```py
print(model.get_memory_footprint())
```

يمكن تحميل النماذج المضغوطة من طريقة [`~PreTrainedModel.from_pretrained`] دون الحاجة إلى تحديد معلمات `load_in_8bit` أو `load_in_4bit`:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

## 8-بت (خوارزمية LLM.int8())

<Tip>  

تعرف على المزيد حول تفاصيل الضغط إلى 8 بت في [منشور المدونة](https://huggingface.co/blog/hf-bitsandbytes-integration) هذا!  

</Tip>

يستكشف هذا القسم بعض الميزات المحددة للنماذج المضغوطة إلى 8 بت، مثل التفريغ، وعتبات القيم الشاذة، وتخطي تحويل الوحدات، والضبط الدقيق.

### التفريغ

يمكن للنماذج المضغوطة إلى 8 بت تفريغ الأوزان بين وحدة المعالجة المركزية ووحدة معالجة الرسومات لدعم تثبيت نماذج كبيرة جدًا في الذاكرة. يتم تخزين الأوزان المرسلة إلى وحدة المعالجة المركزية فعليًا في **float32**، ولا يتم تحويلها إلى 8 بت. على سبيل المثال، لتمكين التفريغ لنموذج [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)، ابدأ بإنشاء [`BitsAndBytesConfig`]:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

صمم خريطة أجهزة مخصصة لتناسب كل شيء على وحدة معالجة الرسومات الخاصة بك باستثناء `lm_head`، والتي سترسلها إلى وحدة المعالجة المركزية:

```py
device_map = {
"transformer.word_embeddings": 0,
"transformer.word_embeddings_layernorm": 0,
"lm_head": "cpu",
"transformer.h": 0,
"transformer.ln_f": 0,
}
```

الآن قم بتحميل نموذجك باستخدام `device_map` و`quantization_config` المخصصين:

```py
model_8bit = AutoModelForCausalLM.from_pretrained(
"bigscience/bloom-1b7",
device_map=device_map,
quantization_config=quantization_config,
)
```

### عتبة القيم الشاذة

"القيمة الشاذة" هي قيمة حالة مخفية أكبر من عتبة معينة، ويتم حساب هذه القيم في fp16. في حين أن القيم تكون عادةً موزعة بشكل طبيعي ([-3.5، 3.5])، فقد يكون هذا التوزيع مختلفًا جدًا للنماذج الكبيرة ([-60، 6] أو [6، 60]). يعمل الضغط إلى 8 بت بشكل جيد للقيم ~5، ولكن بعد ذلك، هناك عقوبة أداء كبيرة. العتبة الافتراضية الجيدة هي 6، ولكن قد تكون هناك حاجة إلى عتبة أقل للنماذج الأقل استقرارًا (النماذج الصغيرة أو الضبط الدقيق).

لإيجاد أفضل عتبة لنموذجك، نوصي بالتجربة مع معلمة `llm_int8_threshold` في [`BitsAndBytesConfig`]:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
model_id,
device_map=device_map,
quantization_config=quantization_config,
)
```

### تخطي تحويل الوحدات

بالنسبة لبعض النماذج، مثل [Jukebox](model_doc/jukebox)، لا تحتاج إلى ضغط كل وحدة إلى 8 بت، بل يمكن أن يتسبب ذلك في عدم الاستقرار. مع Jukebox، هناك عدة وحدات `lm_head` يجب تخطيها باستخدام معلمة `llm_int8_skip_modules` في [`BitsAndBytesConfig`]:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
model_id,
device_map="auto",
quantization_config=quantization_config,
)
```

### الضبط الدقيق

مع مكتبة [PEFT](https://github.com/huggingface/peft)، يمكنك الضبط الدقيق لنماذج كبيرة مثل [flan-t5-large](https://huggingface.co/google/flan-t5-large) و[facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) مع الضغط إلى 8 بت. لا تحتاج إلى تمرير معلمة `device_map` للتدريب لأنه سيقوم تلقائيًا بتحميل نموذجك على وحدة معالجة الرسومات. ومع ذلك، يمكنك لا تزال تخصيص خريطة الجهاز باستخدام معلمة `device_map` إذا كنت تريد ذلك (`device_map="auto"` يجب أن تستخدم فقط للاستدلال).

## 4-بت (خوارزمية QLoRA)

<Tip>  

جرب الضغط إلى 4 بت في هذا [دفتر الملاحظات](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf) وتعرف على المزيد حول تفاصيله في [منشور المدونة](https://huggingface.co/blog/4bit-transformers-bitsandbytes) هذا.  

</Tip>

يستكشف هذا القسم بعض الميزات المحددة للنماذج المضغوطة إلى 4 بت، مثل تغيير نوع بيانات الحساب، واستخدام نوع بيانات Normal Float 4 (NF4)، واستخدام الضغط المتداخل.

### نوع بيانات الحساب

لتسريع الحساب، يمكنك تغيير نوع البيانات من float32 (القيمة الافتراضية) إلى bf16 باستخدام معلمة `bnb_4bit_compute_dtype` في [`BitsAndBytesConfig`]:

```py
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```
### التعويم العادي 4 (NF4)
NF4 هو نوع بيانات مكون من 4 بتات من الورقة البحثية [QLoRA](https://hf.co/papers/2305.14314)، تم تكييفه مع الأوزان المُحَمَّلة من توزيع عادي. يجب استخدام NF4 لتدريب النماذج الأساسية المكونة من 4 بتات. يمكن تهيئتها باستخدام معلمة `bnb_4bit_quant_type` في [`BitsAndBytesConfig`]:
```py
من transformers استورد BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

بالنسبة للاستنتاج، لا يكون لـ `bnb_4bit_quant_type` تأثير كبير على الأداء. ومع ذلك، للحفاظ على الاتساق مع أوزان النموذج، يجب استخدام قيم `bnb_4bit_compute_dtype` و `torch_dtype`.

### التكميم المُعشَّش
التكميم المُعشَّش هو تقنية يمكن أن توفر ذاكرة إضافية دون تكلفة أداء إضافية. تقوم هذه الميزة بتنفيذ تكميم ثانٍ للأوزان المُكَمَّمة بالفعل لتوفير 0.4 بت/معلمة إضافية. على سبيل المثال، باستخدام التكميم المُعشَّش، يمكنك ضبط نموذج [Llama-13b](https://huggingface.co/meta-llama/Llama-2-13b) على معالج رسومات NVIDIA T4 GPU بسعة 16 جيجابايت مع طول تسلسل يبلغ 1024، وحجم دفعة يبلغ 1، وتمكين تراكم التدرجات مع 4 خطوات.

```py
من transformers استورد BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b"، quantization_config=double_quant_config)
```

## فك تكميم نماذج `bitsandbytes`
بمجرد تكميم النموذج، يمكنك فك تكميمه إلى الدقة الأصلية، ولكن قد يؤدي ذلك إلى فقدان صغير في جودة النموذج. تأكد من أن لديك ذاكرة وصول عشوائي (RAM) كافية في وحدة معالجة الرسومات (GPU) لتناسب النموذج بعد فك تكميمه.

```python
من transformers استورد AutoModelForCausalLM و BitsAndBytesConfig و AutoTokenizer

model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id, BitsAndBytesConfig(load_in_4bit=True))
tokenizer = AutoTokenizer.from_pretrained(model_id)

قم بفك تكميم النموذج

النص = tokenizer ("مرحبا اسمي هو"، return_tensors="pt").to(0)

out = model.generate(**text)
print(tokenizer.decode(out[0]))
```