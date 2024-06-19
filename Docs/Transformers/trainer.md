# Trainer

تتيح وحدة [`Trainer`] حلقة تدريب وتقييم كاملة لنماذج PyTorch المنفذة في مكتبة Transformers. كل ما عليك فعله هو تمرير القطع اللازمة للتدريب (مثل النموذج، ومعالج التحليل اللغوي، ومجموعة البيانات، ووظيفة التقييم، وفرط معلمات التدريب، وما إلى ذلك)، وستتولى فئة [`Trainer`] الباقي. يسهل ذلك بدء التدريب بشكل أسرع دون الحاجة إلى كتابة حلقة التدريب الخاصة بك يدويًا. وفي الوقت نفسه، تتميز وحدة [`Trainer`] بإمكانية تخصيصها بشكل كبير، وتوفر العديد من خيارات التدريب حتى تتمكن من تكييفها مع متطلبات التدريب الخاصة بك بالضبط.

<Tip>
بالإضافة إلى فئة [`Trainer`], توفر مكتبة Transformers أيضًا فئة [`Seq2SeqTrainer`] للمهام التسلسلية إلى التسلسلية مثل الترجمة أو الموجز. هناك أيضًا فئة [`~trl.SFTTrainer`] من مكتبة [TRL](https://hf.co/docs/trl) التي تغلف فئة [`Trainer`] والتي تم تحسينها لتدريب نماذج اللغة مثل Llama-2 وMistral باستخدام تقنيات التوليد المتسلسل. كما يدعم [`~trl.SFTTrainer`] ميزات مثل تعبئة التسلسل، وLoRA، والكمية، وDeepSpeed للتحجيم بكفاءة إلى أي حجم نموذج.
<br>
لا تتردد في الاطلاع على [مرجع API](./main_classes/trainer) لفئات [`Trainer`]-type الأخرى هذه لمعرفة المزيد حول متى يجب استخدام كل منها. بشكل عام، تعد [`Trainer`] أكثر الخيارات تنوعًا وهي مناسبة لمجموعة واسعة من المهام. تم تصميم [`Seq2SeqTrainer`] للمهام التسلسلية إلى التسلسلية، وتم تصميم [`~trl.SFTTrainer`] لتدريب نماذج اللغة.
</Tip>

قبل البدء، تأكد من تثبيت [Accelerate](https://hf.co/docs/accelerate) - وهي مكتبة لتمكين وتشغيل التدريب على PyTorch عبر بيئات موزعة.

```bash
pip install accelerate

# upgrade
pip install accelerate --upgrade
```

يوفر هذا الدليل نظرة عامة على فئة [`Trainer`].

## الاستخدام الأساسي

تتضمن وحدة [`Trainer`] جميع التعليمات البرمجية التي ستجدها في حلقة تدريب أساسية:

1. قم بتنفيذ خطوة تدريب لحساب الخسارة
2. احسب المشتقات باستخدام طريقة [`~accelerate.Accelerator.backward`]
3. تحديث الأوزان بناءً على المشتقات
4. كرر هذه العملية حتى تصل إلى عدد محدد مسبقًا من العصور

تفصل فئة [`Trainer`] كل هذه التعليمات البرمجية بحيث لا يتعين عليك القلق بشأن كتابة حلقة تدريب يدويًا في كل مرة أو إذا كنت بدأت للتو في PyTorch والتدريب. كل ما عليك فعله هو توفير المكونات الأساسية اللازمة للتدريب، مثل النموذج ومجموعة البيانات، وستتعامل فئة [`Trainer`] مع كل شيء آخر.

إذا كنت تريد تحديد أي خيارات تدريب أو فرط معلمات، فيمكنك العثور عليها في فئة [`TrainingArguments`]. على سبيل المثال، دعنا نحدد أين سيتم حفظ النموذج في `output_dir` ودفع النموذج إلى Hub بعد التدريب باستخدام `push_to_hub=True`.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
output_dir="your-model",
learning_rate=2e-5,
per_device_train_batch_size=16,
per_device_eval_batch_size=16,
num_train_epochs=2,
weight_decay=0.01,
eval_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
push_to_hub=True,
)
```

مرر `training_args` إلى [`Trainer`] بالإضافة إلى نموذج، ومجموعة بيانات، وشيء لمعالجة مجموعة البيانات (اعتمادًا على نوع البيانات، قد يكون معالجًا للتحليل اللغوي أو مستخرجًا للميزات أو معالجًا للصور)، ومجمع بيانات، ووظيفة لحساب المقاييس التي تريد تتبعها أثناء التدريب.

أخيرًا، اتصل بوظيفة [`~Trainer.train`] لبدء التدريب!

```py
from transformers import Trainer

trainer = Trainer(
model=model,
args=training_args,
train_dataset=dataset["train"],
eval_dataset=dataset["test"],
tokenizer=tokenizer,
data_collator=data_collator,
compute_metrics=compute_metrics,
)

trainer.train()
```

### نقاط المراقبة

تحفظ فئة [`Trainer`] نقاط مراقبة النموذج في الدليل المحدد في معلمة `output_dir` من [`TrainingArguments`]. ستجد نقاط المراقبة المحفوظة في مجلد فرعي يسمى `checkpoint-000` حيث تتوافق الأرقام في النهاية مع خطوة التدريب. إن حفظ نقاط المراقبة مفيد لاستئناف التدريب لاحقًا.

```py
# resume from latest checkpoint
trainer.train(resume_from_checkpoint=True)

# resume from specific checkpoint saved in output directory
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

يمكنك حفظ نقاط المراجعة الخاصة بك (لا يتم حفظ حالة المحسن بشكل افتراضي) إلى Hub عن طريق تعيين `push_to_hub=True` في [`TrainingArguments`] لارتكابها ودفعها. والخيارات الأخرى لاتخاذ القرار بشأن كيفية حفظ نقاط المراقبة الخاصة بك هي الإعداد في معلمة [`hub_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_strategy):

* `hub_strategy="checkpoint"` يدفع أحدث نقطة مراقبة إلى مجلد فرعي يسمى "last-checkpoint" يمكنك استئناف التدريب منه
* `hub_strategy="all_checkpoints"` يدفع جميع نقاط المراقبة إلى الدليل المحدد في `output_dir` (سترى نقطة مراقبة واحدة لكل مجلد في مستودع النموذج الخاص بك)

عندما تستأنف التدريب من نقطة مراقبة، تحاول وحدة [`Trainer`] الحفاظ على حالات RNG Python وNumPy وPyTorch كما كانت عندما تم حفظ نقطة المراقبة. ولكن لأن PyTorch لديها العديد من الإعدادات الافتراضية غير الحتمية، فإن حالات RNG غير مضمونة لتكون هي نفسها. إذا كنت تريد تمكين الحتمية الكاملة، فراجع دليل [Controlling sources of randomness](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness) لمعرفة ما يمكنك تمكينه لجعل تدريبك حتميًا تمامًا. ضع في اعتبارك أنه من خلال جعل إعدادات معينة حتمية، قد يكون التدريب أبطأ.

## تخصيص المدرب

في حين أن فئة [`Trainer`] مصممة لتكون سهلة الوصول والاستخدام، إلا أنها توفر أيضًا الكثير من قابلية التخصيص للمستخدمين المغامرين. يمكن أن تكون العديد من طرق [`Trainer`] فئة فرعية وتكون متجاوزة لدعم الوظائف التي تريدها، دون الحاجة إلى إعادة كتابة حلقة التدريب بأكملها من الصفر لاستيعابها. تتضمن هذه الطرق ما يلي:

* [`~Trainer.get_train_dataloader`] ينشئ DataLoader تدريب
* [`~Trainer.get_eval_dataloader`] ينشئ DataLoader تقييم
* [`~Trainer.get_test_dataloader`] ينشئ DataLoader اختبار
* [`~Trainer.log`] يسجل معلومات حول مختلف الكائنات التي تراقب التدريب
* [`~Trainer.create_optimizer_and_scheduler`] ينشئ محسنًا ومخططًا لمعدل التعلم إذا لم يتم تمريرهما في `__init__`؛ يمكن أيضًا تخصيص هذه الوظائف بشكل منفصل باستخدام [`~Trainer.create_optimizer`] و [`~Trainer.create_scheduler`] على التوالي
* [`~Trainer.compute_loss`] يحسب الخسارة على دفعة من إدخالات التدريب
* [`~Trainer.training_step`] يؤدي خطوة التدريب
* [`~Trainer.prediction_step`] يؤدي خطوة التنبؤ والاختبار
* [`~Trainer.evaluate`] يقيم النموذج ويعيد مقاييس التقييم
* [`~Trainer.predict`] يجعل التنبؤات (مع المقاييس إذا كانت العلامات متاحة) على مجموعة الاختبار

على سبيل المثال، إذا كنت تريد تخصيص طريقة [`~Trainer.compute_loss`] لاستخدام خسارة مرجحة بدلاً من ذلك.

```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
def compute_loss(self, model, inputs, return_outputs=False):
labels = inputs.pop("labels")
# forward pass
outputs = model(**inputs)
logits = outputs.get("logits")
# compute custom loss for 3 labels with different weights
loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
return (loss, outputs) if return_outputs else loss
```

### الاستدعاءات

خيار آخر لتخصيص وحدة [`Trainer`] هو استخدام [الاستدعاءات](callbacks). لا تغير الاستدعاءات *أي شيء* في حلقة التدريب. إنهم يفحصون حالة حلقة التدريب ثم ينفذون بعض الإجراءات (مثل التوقف المبكر أو تسجيل النتائج، وما إلى ذلك) اعتمادًا على الحالة. وبعبارة أخرى، لا يمكن استخدام الاستدعاء لتنفيذ شيء مثل وظيفة خسارة مخصصة، ويجب عليك إنشاء فئة فرعية لتجاوز طريقة [`~Trainer.compute_loss`] لذلك.

على سبيل المثال، إذا كنت تريد إضافة استدعاء إيقاف مبكر إلى حلقة التدريب بعد 10 خطوات.

```py
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
def __init__(self, num_steps=10):
self.num_steps = num_steps

def on_step_end(self, args, state, control, **kwargs):
if state.global_step >= self.num_steps:
return {"should_training_stop": True}
else:
return {}
```

ثم مرره إلى معلمة `callback` في [`Trainer`].

```py
from transformers import Trainer

trainer = Trainer(
model=model,
args=training_args,
train_dataset=dataset["train"],
eval_dataset=dataset["test"],
tokenizer=tokenizer,
data_collator=data_collator,
compute_metrics=compute_metrics,
callback=[EarlyStoppingCallback()],
)
```
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين.

## التسجيل

يتم تعيين [`Trainer`] إلى `logging.INFO` بشكل افتراضي، والذي يقوم بالإبلاغ عن الأخطاء والتحذيرات وغيرها من المعلومات الأساسية. يتم تعيين نسخة [`Trainer`] - في البيئات الموزعة - إلى `logging.WARNING`، والتي تقوم بالإبلاغ عن الأخطاء والتحذيرات فقط. يمكنك تغيير مستوى التسجيل باستخدام معلمات [`log_level`] و [`log_level_replica`] في [`TrainingArguments`].

لتهيئة إعداد مستوى السجل لكل عقدة، استخدم معلمة [`log_on_each_node`] لتحديد ما إذا كنت تريد استخدام مستوى السجل على كل عقدة أو فقط على العقدة الرئيسية.

يحدد [`Trainer`] مستوى السجل بشكل منفصل لكل عقدة في طريقة [`Trainer.__init__`]، لذا قد ترغب في النظر في تعيين هذا الأمر في وقت سابق إذا كنت تستخدم وظائف Transformers الأخرى قبل إنشاء كائن [`Trainer`].

على سبيل المثال، لتعيين رمزك الأساسي ووحداتك النمطية لاستخدام نفس مستوى السجل وفقًا لكل عقدة:

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
datefmt="%m/%d/%Y %H:%M:%S",
handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

استخدم تركيبات مختلفة من `log_level` و`log_level_replica` لتهيئة ما يتم تسجيله على كل من العقد.

## NEFTune

[NEFTune] (https://hf.co/papers/2310.05914) هي تقنية يمكنها تحسين الأداء عن طريق إضافة ضوضاء إلى متجهات التضمين أثناء التدريب. لتمكينه في [`Trainer`)، قم بتعيين معلمة` neftune_noise_alpha` في [`TrainingArguments`] للتحكم في مقدار الضوضاء المضافة.

```بي
من المحولات استيراد TrainingArguments، المدرب

training_args = TrainingArguments (..., neftune_noise_alpha = 0.1)
المدرب = مدرب (..., args = training_args)
```

يتم تعطيل NEFTune بعد التدريب لاستعادة طبقة التضمين الأصلية لتجنب أي سلوك غير متوقع.

## GaLore

Gradient Low-Rank Projection (GaLore) هي استراتيجية تدريب فعالة من حيث الذاكرة منخفضة الرتبة تسمح بالتعلم الكامل للبارامترات ولكنها أكثر كفاءة من حيث الذاكرة من أساليب التكيف منخفضة الرتبة الشائعة، مثل LoRA.

أولاً، تأكد من تثبيت المستودع الرسمي لـ GaLore:

```bash
pip install galore-torch
```

ثم قم ببساطة بإضافة واحد من `["galore_adamw"، "galore_adafactor"، "galore_adamw_8bit"]` في `optim` جنبًا إلى جنب مع `optim_target_modules`، والتي يمكن أن تكون قائمة من السلاسل أو regex أو المسار الكامل المقابل لأسماء الوحدات النمطية المستهدفة التي تريد تكييفها. فيما يلي مثال على نص البرنامج النصي (تأكد من `pip install trl datasets`):

```python
استيراد الشعلة
استيراد مجموعات البيانات
استيراد ترل

من المحولات استيراد TrainingArguments، AutoConfig، AutoTokenizer، AutoModelForCausalLM

train_dataset = datasets.load_dataset ('imdb', split = 'train')

args = TrainingArguments (
output_dir = "./test-galore"،
max_steps = 100،
per_device_train_batch_size = 2،
optim = "galore_adamw"،
optim_target_modules = ["attn"، "mlp"]
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained (model_id)

tokenizer = AutoTokenizer.from_pretrained (model_id)
model = AutoModelForCausalLM.from_config (config).to (0)

المدرب = trl.SFTTrainer (
نموذج = النموذج،
args = args،
train_dataset = train_dataset،
dataset_text_field = 'text'،
max_seq_length = 512،
)

المدرب. train ()
```

لإرسال حجج إضافية تدعمها GaLore، يجب عليك تمرير `optim_args` بشكل صحيح، على سبيل المثال:

```python
استيراد الشعلة
استيراد مجموعات البيانات
استيراد ترل

من المحولات استيراد TrainingArguments، AutoConfig، AutoTokenizer، AutoModelForCausalLM

train_dataset = datasets.load_dataset ('imdb', split = 'train')

args = TrainingArguments (
output_dir = "./test-galore"،
max_steps = 100،
per_device_train_batch_size = 2،
optim = "galore_adamw"،
optim_target_modules = ["attn"، "mlp"]،
optim_args = "rank = 64، update_proj_gap = 100، scale = 0.10"،
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained (model_id)

tokenizer = AutoTokenizer.from_pretrained (model_id)
model = AutoModelForCausalLM.from_config (config).to (0)

المدرب = trl.SFTTrainer (
نموذج = النموذج،
args = args،
train_dataset = train_dataset،
dataset_text_field = 'text'،
max_seq_length = 512،
)

المدرب. train ()
```

يمكنك قراءة المزيد حول الطريقة في [المستودع الأصلي] (https://github.com/jiaweizzhao/GaLore) أو [الورقة] (https://arxiv.org/abs/2403.03507).

حاليًا، يمكنك فقط تدريب الطبقات الخطية التي تعتبر طبقات GaLore وستستخدم التحلل من الرتبة المنخفضة للتدريب بينما سيتم تحسين الطبقات المتبقية بالطريقة التقليدية.

لاحظ أنه سيستغرق بعض الوقت قبل بدء التدريب (~ 3 دقائق لنموذج 2B على NVIDIA A100)، ولكن يجب أن يسير التدريب بسلاسة بعد ذلك.

يمكنك أيضًا إجراء تحسين الطبقة عن طريق إضافة لاحقة `layerwise` إلى اسم المحسن كما هو موضح أدناه:

```python
استيراد الشعلة
استيراد مجموعات البيانات
استيراد ترل

من المحولات استيراد TrainingArguments، AutoConfig، AutoTokenizer، AutoModelForCausalLM

train_dataset = datasets.load_dataset ('imdb', split = 'train')

args = TrainingArguments (
output_dir = "./test-galore"،
max_steps = 100،
per_device_train_batch_size = 2،
optim = "galore_adamw_layerwise"،
optim_target_modules = ["attn"، "mlp"]
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained (model_id)

tokenizer = AutoTokenizer.from_pretrained (model_id)
model = AutoModelForCausalLM.from_config (config).to (0)

المدرب = trl.SFTTrainer (
نموذج = النموذج،
args = args،
train_dataset = train_dataset،
dataset_text_field = 'text'،
max_seq_length = 512،
)

المدرب. train ()
```

لاحظ أن تحسين الطبقة تجريبي إلى حد ما ولا يدعم DDP (Distributed Data Parallel)، لذلك يمكنك تشغيل نص البرنامج النصي للتدريب على وحدة GPU واحدة فقط. يرجى الاطلاع على [هذا القسم المناسب] (https://github.com/jiaweizzhao/GaLore؟tab=readme-ov-file # train-7b-model-with-a-single-gpu-with-24gb-memory) لمزيد من التفاصيل. قد لا تكون الميزات الأخرى مثل قصاصة التدرج أو DeepSpeed مدعومة بشكل افتراضي. يرجى [إثارة مشكلة على GitHub] (https://github.com/huggingface/transformers/issues) إذا واجهت مثل هذه المشكلة.

## محسن LOMO

تم تقديم محسنات LOMO في [التدريب الدقيق للبارامترات الكاملة لنماذج اللغة الكبيرة باستخدام موارد محدودة] (https://hf.co/papers/2306.09782) و [AdaLomo: Low-memory Optimization with Adaptive Learning Rate] (https://hf.co/papers/2310.10195).

يتكون كلاهما من طريقة تدريب دقيقة للبارامترات الكاملة تتسم بالكفاءة. تدمج المحسنات LOMO حساب التدرج وتحديث البارامترات في خطوة واحدة لتقليل استخدام الذاكرة. المحسنات المدعومة لـ LOMO هي "lomo" و "adalomo". أولاً قم بتثبيت LOMO إما من pypi `pip install lomo-optim` أو قم بتثبيته من المصدر باستخدام `pip install git+https://github.com/OpenLMLab/LOMO.git`.

> نصيحة
>
> وفقًا للمؤلفين، يوصى باستخدام `AdaLomo` بدون `grad_norm` للحصول على أداء أفضل وسرعة أعلى.

فيما يلي نص برمجي بسيط يوضح كيفية ضبط دقة [google/gemma-2b] (https://huggingface.co/google/gemma-2b) على مجموعة بيانات IMDB في الدقة الكاملة:

```python
استيراد الشعلة
استيراد مجموعات البيانات
من المحولات استيراد TrainingArguments، AutoTokenizer، AutoModelForCausalLM
استيراد ترل

train_dataset = datasets.load_dataset ('imdb', split = 'train')

args = TrainingArguments (
output_dir = "./test-lomo"،
max_steps = 1000،
per_device_train_batch_size = 4،
optim = "adalomo"،
gradient_checkpointing = True،
logging_strategy = "steps"،
logging_steps = 1،
learning_rate = 2e-6،
save_strategy = "no"،
run_name = "lomo-imdb"،
)

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained (model_id)
model = AutoModelForCausalLM.from_pretrained (model_id، low_cpu_mem_usage = True).to (0)

المدرب = trl.SFTTrainer (
نموذج = النموذج،
args = args،
train_dataset = train_dataset،
dataset_text_field = 'text'،
max_seq_length = 1024،
)

المدرب. train ()
```
## تسريع وتدريب 

تعتمد فئة [Trainer] على مكتبة Accelerate، وهي مكتبة لتسهيل تدريب نماذج PyTorch في بيئات موزعة مع دعم التكاملات مثل FullyShardedDataParallel (FSDP) وDeepSpeed. 

لاستخدام Accelerate مع [Trainer]، قم بتشغيل أمر accelerate.config لإعداد التدريب لبيئة التدريب الخاصة بك. يقوم هذا الأمر بإنشاء ملف config_file.yaml الذي سيتم استخدامه عند تشغيل نص التدريب الخاص بك. على سبيل المثال، بعض تكوينات المثال التي يمكنك إعدادها هي: 

لتشغيل نص التدريب run_glue.py باستخدام تكوين FSDP: 

```bash
accelerate launch \
./examples/pytorch/text-classification/run_glue.py \
--model_name_or_path google-bert/bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

يمكنك أيضًا تحديد المعلمات من ملف config_file.yaml مباشرة في سطر الأوامر: 

```bash
accelerate launch --num_processes=2 \
--use_fsdp \
--mixed_precision=bf16 \
--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
--fsdp_transformer_layer_cls_to_wrap="BertLayer" \
--fsdp_sharding_strategy=1 \
--fsdp_state_dict_type=FULL_STATE_DICT \
./examples/pytorch/text-classification/run_glue.py \
--model_name_or_path google-bert/bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

راجع البرنامج التعليمي "تشغيل البرامج النصية الخاصة بك Accelerate" لمعرفة المزيد حول accelerate_launch والتكوينات المخصصة.