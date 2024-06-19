## التعرف التلقائي على الكلام

يحول التعرف التلقائي على الكلام إشارة الكلام إلى نص، عن طريق رسم خريطة لسلسلة من المدخلات الصوتية إلى مخرجات نصية. يستخدم المساعدون الافتراضيون مثل Siri وAlexa نماذج التعرف على الكلام لمساعدة المستخدمين يوميًا، وهناك العديد من التطبيقات المفيدة الأخرى التي تتفاعل مع المستخدم مثل الترجمة الفورية وتدوين الملاحظات أثناء الاجتماعات.

سيوضح هذا الدليل لك كيفية:

1. ضبط نموذج Wav2Vec2 الدقيق على مجموعة بيانات MInDS-14 لنسخ الصوت إلى نص.
2. استخدام النموذج الدقيق للاستنتاج.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate jiwer
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات MInDS-14

ابدأ بتحميل جزء فرعي أصغر من مجموعة بيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) من مكتبة Datasets 🤗. سيعطيك هذا فرصة لتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

قسِّم مجموعة البيانات إلى مجموعة بيانات للتدريب وأخرى للاختبار باستخدام طريقة [`~Dataset.train_test_split`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.train_test_split):

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

ثم الق نظرة على مجموعة البيانات:

```py
>>> minds
DatasetDict({
train: Dataset({
features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
num_rows: 16
})
test: Dataset({
features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
num_rows: 4
})
})
```

على الرغم من أن مجموعة البيانات تحتوي على الكثير من المعلومات المفيدة، مثل `lang_id` و`english_transcription`، إلا أنك ستركز على `audio` و`transcription` في هذا الدليل. قم بإزالة الأعمدة الأخرى باستخدام طريقة [`~datasets.Dataset.remove_columns`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.remove_columns):

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

الق نظرة على المثال مرة أخرى:

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
0.00024414,  0.00024414], dtype=float32),
'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
'sampling_rate': 8000},
'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

هناك حقلان:

- `audio`: مصفوفة أحادية البعد لإشارة الكلام التي يجب استدعاؤها لتحميل وإعادة أخذ عينات ملف الصوت.
- `transcription`: النص المستهدف.

## معالجة مسبقة

الخطوة التالية هي تحميل معالج Wav2Vec2 لمعالجة إشارة الصوت:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

تحتوي مجموعة بيانات MInDS-14 على معدل أخذ عينات يبلغ 8000 كيلو هرتز (يمكنك العثور على هذه المعلومات في [بطاقة مجموعة البيانات](https://huggingface.co/datasets/PolyAI/minds14) الخاصة بها)، مما يعني أنك ستحتاج إلى إعادة أخذ عينات من مجموعة البيانات إلى 16000 كيلو هرتز لاستخدام نموذج Wav2Vec2 المُدرَّب مسبقًا:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
'sampling_rate': 16000},
'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

كما هو موضح في `transcription` أعلاه، يحتوي النص على مزيج من الأحرف الكبيرة والصغيرة. تم تدريب معجم Wav2Vec2 على الأحرف الكبيرة فقط، لذلك ستحتاج إلى التأكد من أن النص يتطابق مع مفردات المعجم:

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

الآن قم بإنشاء دالة معالجة مسبقة تقوم بما يلي:

1. استدعاء عمود `audio` لتحميل وإعادة أخذ عينات ملف الصوت.
2. استخراج `input_values` من ملف الصوت ورقمنة عمود `transcription` باستخدام المعالج.

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) في مكتبة Datasets 🤗. يمكنك تسريع `map` عن طريق زيادة عدد العمليات باستخدام معلمة `num_proc`:

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

لا تحتوي مكتبة 🤗 Transformers على جامع بيانات للتعرف التلقائي على الكلام، لذلك ستحتاج إلى تكييف [`DataCollatorWithPadding`](https://huggingface.co/docs/transformers/main_classes/data_collator) لإنشاء دفعة من الأمثلة. كما أنه سيقوم أيضًا بتبطين ديناميكي لنصك وتسمياتك إلى طول العنصر الأطول في دفعتها (بدلاً من مجموعة البيانات بأكملها) بحيث يكون لها طول موحد. في حين أنه من الممكن تبطين نصك في دالة `tokenizer` عن طريق تعيين `padding=True`، فإن التبطين الديناميكي أكثر كفاءة.

على عكس جامعي البيانات الآخرين، يحتاج جامع البيانات هذا إلى تطبيق طريقة تبطين مختلفة على `input_values` و`labels`:

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

الآن قم بتنفيذ جامع بيانات `DataCollatorForCTCWithPadding`:

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة [Evaluate](https://huggingface.co/docs/evaluate/index) 🤗. بالنسبة لهذه المهمة، قم بتحميل مقياس [خطأ كلمة](https://huggingface.co/spaces/evaluate-metric/wer) (WER) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`](https://huggingface.co/docs/evaluate/main_classes/evaluation_module#computemetrics) لحساب WER:

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

دالتك `compute_metrics` جاهزة الآن، وستعود إليها عند إعداد التدريب الخاص بك.
بالتأكيد! فيما يلي ترجمة للنص الموجود في الفقرات والعناوين، مع اتباع التعليمات التي قدمتها:

## التدريب

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`]، فراجع البرنامج التعليمي الأساسي [هنا] (../training#train-with-pytorch-trainer)

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل Wav2Vec2 مع [`AutoModelForCTC`]. حدد التخفيض الذي سيتم تطبيقه باستخدام معلمة `ctc_loss_reduction`. غالبًا ما يكون من الأفضل استخدام المتوسط بدلاً من الجمع الافتراضي:

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد أين سيتم حفظ نموذجك. ستقوم بتحميل هذا النموذج إلى المحور عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم WER وحفظ نقطة تفتيش التدريب.

2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمحلل اللغوي ومجمع البيانات ووظيفة `compute_metrics`.

3. اتصل [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model"،
...     per_device_train_batch_size=8،
...     gradient_accumulation_steps=2،
...     learning_rate=1e-5،
...     warmup_steps=500،
...     max_steps=2000،
...     gradient_checkpointing=True،
...     fp16=True،
...     group_by_length=True،
...     eval_strategy="steps"،
...     per_device_eval_batch_size=8،
...     save_steps=1000،
...     eval_steps=1000،
...     logging_steps=25،
...     load_best_model_at_end=True،
...     metric_for_best_model="wer"،
...     greater_is_better=False،
...     push_to_hub=True،
... )

>>> trainer = Trainer(
...     model=model،
...     args=training_args،
...     train_dataset=encoded_minds["train"]،
...     eval_dataset=encoded_minds["test"]،
...     tokenizer=processor،
...     data_collator=data_collator،
...     compute_metrics=compute_metrics،
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك في المحور باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```

للحصول على مثال أكثر تفصيلاً حول كيفية ضبط نموذج للتعرف التلقائي على الكلام، راجع هذه المدونة [المنشور] (https://huggingface.co/blog/fine-tune-wav2vec2-english) للتعرف على الكلام باللغة الإنجليزية وهذا [المنشور] (https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) للتعرف على الكلام متعدد اللغات.

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

قم بتحميل ملف صوتي تريد تشغيل الاستدلال عليه. تذكر إعادة أخذ معدل عينة معدل عينة ملف الصوت لمطابقة معدل عينة النموذج إذا لزم الأمر!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه في [`pipeline`]. قم بتنفيذ خط أنابيب للتعرف التلقائي على الكلام باستخدام نموذجك، ومرر ملف الصوت الخاص بك إليه:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE TO SET UP JOINT ACCOUNT WITH MY PARTNER'}
```

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا أردت:

قم بتحميل معالج لمعالجة ملف الصوت والنسخ النصي وإرجاع الإدخال كموترات PyTorch:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

مرر إدخالاتك إلى النموذج وأعد النتائج:

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على `input_ids` المتوقع مع أعلى احتمال، واستخدم المعالج لترميز `input_ids` المتوقع مرة أخرى إلى نص:

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOULD LIKE TO SET UP JOINT ACCOUNT WITH MY PARTNER']
```