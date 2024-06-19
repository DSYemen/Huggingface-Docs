# GPTQ

مكتبة AutoGPTQ تنفذ خوارزمية GPTQ، وهي تقنية لضغط النموذج بعد التدريب حيث يتم ضغط كل صف في مصفوفة الأوزان بشكل مستقل لإيجاد إصدار من الأوزان التي تقلل من الخطأ. يتم ضغط هذه الأوزان إلى int4، ولكن يتم استعادتها إلى fp16 أثناء الاستدلال. يمكن أن يوفر هذا استخدام الذاكرة الخاصة بك بمقدار 4x لأن أوزان int4 يتم إلغاء ضغطها في نواة مدمجة بدلاً من الذاكرة العالمية لوحدة معالجة الرسومات (GPU)، ويمكنك أيضًا توقع تسريع الاستدلال لأن استخدام عرض نطاق ترددي أقل يستغرق وقتًا أقل في التواصل.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```bash
pip install auto-gptq
pip install --upgrade accelerate optimum transformers
```

لضغط نموذج (مدعوم حاليًا للنصوص فقط)، يلزمك إنشاء فئة `GPTQConfig` وتعيين عدد البتات التي تريد ضغطها، ومجموعة بيانات لضبط أوزان الضغط، ومعالج لتحضير مجموعة البيانات.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

يمكنك أيضًا تمرير مجموعة البيانات الخاصة بك كقائمة من السلاسل النصية، ولكن يوصى بشدة باستخدام نفس مجموعة البيانات من ورقة GPTQ.

```py
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
```

قم بتحميل نموذج لضغطه ومرر `gptq_config` إلى طريقة `~AutoModelForCausalLM.from_pretrained`. قم بتعيين `device_map="auto"` لنقل النموذج تلقائيًا إلى وحدة المعالجة المركزية (CPU) للمساعدة في تثبيت النموذج في الذاكرة، والسماح بنقل وحدات النموذج بين وحدة المعالجة المركزية ووحدة معالجة الرسومات (GPU) للضغط.

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```

إذا كنت تواجه مشكلة في نفاد الذاكرة لأن مجموعة البيانات كبيرة جدًا، فإن النقل إلى القرص غير مدعوم. إذا كان الأمر كذلك، فحاول تمرير معلمة `max_memory` لتحديد مقدار الذاكرة التي سيتم استخدامها على جهازك (GPU وCPU):

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"}, quantization_config=gptq_config)
```

اعتمادًا على أجهزتك، قد يستغرق ضغط نموذج من الصفر بعض الوقت. قد يستغرق ضغط نموذج facebook/opt-350m حوالي 5 دقائق على وحدة معالجة الرسومات (GPU) من Google Colab من الفئة المجانية، ولكنه سيستغرق حوالي 4 ساعات لضغط نموذج بمعلمات 175B على NVIDIA A100. قبل ضغط النموذج، من الجيد التحقق مما إذا كان هناك بالفعل إصدار مضغوط من GPTQ للنموذج على Hub.

بمجرد ضغط النموذج الخاص بك، يمكنك دفعه إلى Hub مع معالج التحليل حيث يمكن مشاركتها والوصول إليها بسهولة. استخدم طريقة `~PreTrainedModel.push_to_hub` لحفظ `GPTQConfig`:

```py
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

يمكنك أيضًا حفظ نموذجك المضغوط محليًا باستخدام طريقة `~PreTrainedModel.save_pretrained`. إذا تم ضغط النموذج باستخدام معلمة `device_map`، فتأكد من نقل النموذج بالكامل إلى وحدة معالجة الرسومات (GPU) أو وحدة المعالجة المركزية (CPU) قبل حفظه. على سبيل المثال، لحفظ النموذج على وحدة المعالجة المركزية (CPU):

```py
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")

# إذا تم الضغط باستخدام device_map
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

أعد تحميل نموذج مضغوط باستخدام طريقة `~PreTrainedModel.from_pretrained`، وقم بتعيين `device_map="auto"` لتوزيع النموذج تلقائيًا على جميع وحدات معالجة الرسومات (GPU) المتوفرة لتحميل النموذج بشكل أسرع دون استخدام ذاكرة أكثر من اللازم.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

## ExLlama

ExLlama هو تنفيذ Python/C++/CUDA لنموذج Llama مصمم للاستدلال بشكل أسرع باستخدام أوزان GPTQ ذات 4 بتات (تحقق من هذه المعايير). يتم تنشيط نواة ExLlama بشكل افتراضي عند إنشاء كائن `GPTQConfig`. لزيادة سرعة الاستدلال بشكل أكبر، استخدم نواة ExLlamaV2 عن طريق تكوين معلمة `exllama_config`:

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=gptq_config)
```

يتم دعم النماذج ذات 4 بتات فقط، ونوصي بتعطيل نواة ExLlama إذا كنت تقوم بضبط نموذج مضغوط باستخدام PEFT.

تدعم نواة ExLlama فقط عندما يكون النموذج بالكامل على وحدة معالجة الرسومات (GPU). إذا كنت تقوم بالاستدلال على وحدة المعالجة المركزية (CPU) باستخدام AutoGPTQ (الإصدار > 0.4.2)، فستحتاج إلى تعطيل نواة ExLlama. وهذا يكتب فوق السمات المتعلقة بنواة ExLlama في تكوين الضغط لملف config.json.

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig
gptq_config = GPTQConfig(bits=4, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="cpu", quantization_config=gptq_config)
```