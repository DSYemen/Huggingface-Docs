# جولة سريعة

ابدأ استخدام مكتبة 🤗 Transformers! سواء كنت مطورًا أو مستخدمًا عاديًا، ستساعدك هذه الجولة السريعة على البدء وستريك كيفية استخدام [`pipeline`] للاستنتاج، وتحميل نموذج مُدرب مسبقًا ومعالج مسبق باستخدام [AutoClass](./model_doc/auto)، والتدريب السريع لنموذج باستخدام PyTorch أو TensorFlow. إذا كنت مبتدئًا، نوصي بالاطلاع على دروسنا أو [الدورة](https://huggingface.co/course/chapter1/1) للحصول على شرح أكثر تعمقًا للمفاهيم التي تم تقديمها هنا.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
!pip install transformers datasets evaluate accelerate
```

سيتعين عليك أيضًا تثبيت إطار عمل التعلم الآلي المفضل لديك:

<frameworkcontent>
<pt>

```bash
pip install torch
```

</pt>

<tf>

```bash
pip install tensorflow
```

</tf>

</frameworkcontent>

## خط الأنابيب

<Youtube id="tiZFewofSLM"/>

يمثل [`pipeline`] أسهل وأسرع طريقة لاستخدام نموذج مُدرب مسبقًا للاستنتاج. يمكنك استخدام [`pipeline`] جاهزًا للعديد من المهام عبر طرائق مختلفة، والتي يظهر بعضها في الجدول أدناه:

<Tip>

للحصول على قائمة كاملة بالمهام المتاحة، راجع [مرجع API خط الأنابيب](./main_classes/pipelines).

</Tip>

| المهمة | الوصف | الطريقة | معرف خط الأنابيب |
| --- | --- | --- | --- |
| تصنيف النص | تعيين تسمية إلى تسلسل نص معين | NLP | pipeline(task="sentiment-analysis") |
| توليد النص | توليد نص بناءً على موجه | NLP | pipeline(task="text-generation") |
| تلخيص | توليد ملخص لتسلسل نص أو مستند | NLP | pipeline(task="summarization") |
| تصنيف الصور | تعيين تسمية إلى صورة | رؤية حاسوبية | pipeline(task="image-classification") |
| تجزئة الصور | تعيين تسمية إلى كل بكسل في صورة (يدعم التجزئة الدلالية، والكلية، وتجزئة الحالات) | رؤية حاسوبية | pipeline(task="image-segmentation") |
| اكتشاف الأشياء | التنبؤ بصناديق الإحاطة وفئات الأشياء في صورة | رؤية حاسوبية | pipeline(task="object-detection") |
| تصنيف الصوت | تعيين تسمية إلى بيانات صوتية | صوت | pipeline(task="audio-classification") |
| التعرف التلقائي على الكلام | نسخ الكلام إلى نص | صوت | pipeline(task="automatic-speech-recognition") |
| الإجابة على الأسئلة المرئية | الإجابة على سؤال حول الصورة، مع إعطاء صورة وسؤال | متعدد الوسائط | pipeline(task="vqa") |
| الإجابة على أسئلة المستندات | الإجابة على سؤال حول المستند، مع إعطاء مستند وسؤال | متعدد الوسائط | pipeline(task="document-question-answering") |
| وصف الصورة | توليد عنوان لصورة معينة | متعدد الوسائط | pipeline(task="image-to-text") |

ابدأ بإنشاء مثيل من [`pipeline`] وتحديد المهمة التي تريد استخدامه لها. في هذا الدليل، ستستخدم [`pipeline`] لتحليل المشاعر كمثال:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

يقوم [`pipeline`] بتنزيل وتخزين نموذج افتراضي [مُدرب مسبقًا](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) ومعالج للتحليل الدلالي. الآن يمكنك استخدام `classifier` على النص المستهدف:

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

إذا كان لديك أكثر من إدخال واحد، قم بتمرير إدخالاتك كقائمة إلى [`pipeline`] لإرجاع قائمة من القواميس:

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

يمكن لـ [`pipeline`] أيضًا إجراء مسح عبر مجموعة بيانات كاملة لأي مهمة تريدها. في هذا المثال، دعنا نختار التعرف التلقائي على الكلام كمهمة لنا:

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

قم بتحميل مجموعة بيانات صوتية (راجع دليل البدء السريع لـ 🤗 Datasets [هنا](https://huggingface.co/docs/datasets/quickstart#audio) لمزيد من التفاصيل) التي تريد إجراء مسح عبرها. على سبيل المثال، قم بتحميل مجموعة بيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

يجب التأكد من أن معدل أخذ العينات لمجموعة البيانات يتطابق مع معدل أخذ العينات الذي تم تدريب [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) عليه:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

يتم تحميل ملفات الصوت وإعادة أخذ العينات تلقائيًا عند استدعاء العمود `"audio"`.

استخرج صفائف الموجات الصوتية الخام من أول 4 عينات ومررها كقائمة إلى خط الأنابيب:

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I FURN A JOINA COUT']
```

بالنسبة لمجموعات البيانات الأكبر حيث تكون الإدخالات كبيرة (كما هو الحال في الكلام أو الرؤية)، سترغب في تمرير مولد بدلاً من قائمة لتحميل جميع الإدخالات في الذاكرة. راجع [مرجع API خط الأنابيب](./main_classes/pipelines) لمزيد من المعلومات.

### استخدام نموذج ومعالج آخرين في خط الأنابيب

يمكن لـ [`pipeline`] استيعاب أي نموذج من [Hub](https://huggingface.co/models)، مما يجعله سهل التكيف مع حالات استخدام أخرى. على سبيل المثال، إذا كنت تريد نموذجًا قادرًا على التعامل مع النص الفرنسي، فيمكنك استخدام العلامات على Hub لتصفية نموذج مناسب. تعيد النتيجة الأولى المصفاة نموذج BERT [متعدد اللغات](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) الذي تم ضبطه مسبقًا لتحليل المشاعر والذي يمكنك استخدامه للنص الفرنسي:

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>

استخدم [`AutoModelForSequenceClassification`] و [`AutoTokenizer`] لتحميل النموذج المُدرب مسبقًا ومعالجه المرتبط (مزيد من المعلومات حول `AutoClass` في القسم التالي):

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

</pt>

<tf>

استخدم [`TFAutoModelForSequenceClassification`] و [`AutoTokenizer`] لتحميل النموذج المُدرب مسبقًا ومعالجه المرتبط (مزيد من المعلومات حول `TFAutoClass` في القسم التالي):

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

</tf>

</frameworkcontent>

حدد النموذج والمعالج في [`pipeline`]، والآن يمكنك تطبيق `classifier` على النص الفرنسي:

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

إذا لم تتمكن من العثور على نموذج لحالتك الاستخدامية، فسيتعين عليك ضبط نموذج مُدرب مسبقًا على بياناتك. اطلع على [دليل الضبط الدقيق](./training) الخاص بنا لمعرفة كيفية القيام بذلك. وأخيرًا، بعد ضبط نموذجك المُدرب مسبقًا، يرجى التفكير في [مشاركته](./model_sharing) مع المجتمع على Hub لدمقرطة التعلم الآلي للجميع! 🤗
## AutoClass

تعمل الفئتان `AutoModelForSequenceClassification` و `AutoTokenizer` معًا تحت الغطاء لتوفير وظيفة `pipeline` التي استخدمتها أعلاه. تعتبر AutoClass اختصارًا يقوم تلقائيًا باسترداد بنية نموذج مُدرب مسبقًا من اسمه أو مساره. كل ما عليك فعله هو تحديد فئة `AutoClass` المناسبة لمهمتك وفئة ما قبل المعالجة المرتبطة بها.

لنعد إلى المثال من القسم السابق ولنرى كيف يمكنك استخدام فئة `AutoClass` لتكرار نتائج وظيفة `pipeline`.

### AutoTokenizer

تكون أداة التعامل مع الرموز مسؤولة عن معالجة النص مسبقًا إلى مصفوفة من الأرقام كمدخلات لنموذج. هناك قواعد متعددة تحكم عملية التعامل مع الرموز، بما في ذلك كيفية تقسيم الكلمة والمستوى الذي يجب أن تنقسم فيه الكلمات (تعرف المزيد عن التعامل مع الرموز في ملخص أداة التعامل مع الرموز). أهم شيء يجب تذكره هو أنك بحاجة إلى إنشاء مثيل لأداة التعامل مع الرموز بنفس اسم النموذج لضمان استخدامك لقواعد التعامل مع الرموز نفسها التي تم تدريب النموذج عليها مسبقًا.

قم بتحميل أداة التعامل مع الرموز باستخدام `AutoTokenizer`:

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

مرر نصك إلى أداة التعامل مع الرموز:

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

تُرجع أداة التعامل مع الرموز قاموسًا يحتوي على:

- `input_ids`: التمثيلات الرقمية لرموزك.
- `attention_mask`: تشير إلى الرموز التي يجب الاهتمام بها.

يمكن لأداة التعامل مع الرموز أيضًا قبول قائمة من المدخلات، وتقوم بتقسيم النص وتهذيبه لإرجاع دفعة ذات طول موحد:

```py
>>> pt_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="pt",
... )
```

```py
>>> tf_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="tf",
... )
```

### AutoModel

يوفر برنامج 🤗 Transformers طريقة بسيطة وموحدة لتحميل مثيلات مُدربة مسبقًا. وهذا يعني أنه يمكنك تحميل فئة `AutoModel` كما لو كنت تقوم بتحميل فئة `AutoTokenizer`. الفرق الوحيد هو تحديد فئة `AutoModel` الصحيحة للمهمة. بالنسبة لتصنيف النصوص (أو التسلسلات)، يجب عليك تحميل `AutoModelForSequenceClassification`:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

الآن، مرر دفعة المدخلات التي تمت معالجتها مسبقًا مباشرة إلى النموذج. ما عليك سوى فك حزم القاموس عن طريق إضافة `**`:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

يُخرج النموذج التنشيطات النهائية في خاصية `logits`. طبق دالة softmax على `logits` لاسترداد الاحتمالات:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
[0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```

### Save a model

بمجرد ضبط نموذجك، يمكنك حفظه مع أداة التعامل مع الرموز الخاصة به باستخدام `PreTrainedModel.save_pretrained`:

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

عندما تكون مستعدًا لاستخدام النموذج مرة أخرى، أعد تحميله باستخدام `PreTrainedModel.from_pretrained`:

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```

### ميزة رائعة في 🤗 Transformers هي القدرة على حفظ نموذج وإعادة تحميله كنموذج PyTorch أو TensorFlow. يمكن لمعلمة `from_pt` أو `from_tf` تحويل النموذج من إطار عمل إلى آخر:

```py
>>> from transformers import AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained(tf_save_directory)
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(tf_save_directory, from_tf=True)
```

```py
>>> from transformers import TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(pt_save_directory, from_pt=True)
```

## Custom model builds

يمكنك تعديل فئة تكوين النموذج لتغيير طريقة بناء النموذج. يحدد التكوين سمات النموذج، مثل عدد الطبقات المخفية أو رؤوس الاهتمام. تبدأ من الصفر عند تهيئة نموذج من فئة تكوين مخصص. يتم تهيئة سمات النموذج بشكل عشوائي، ويجب تدريب النموذج قبل استخدامه للحصول على نتائج ذات معنى.

ابدأ باستيراد `AutoConfig`، ثم قم بتحميل النموذج المُدرب مسبقًا الذي تريد تعديله. ضمن `AutoConfig.from_pretrained`، يمكنك تحديد السمة التي تريد تغييرها، مثل عدد رؤوس الاهتمام:

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

أنشئ نموذجًا من تكوينك المخصص باستخدام `AutoModel.from_config`:

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```

راجع دليل [إنشاء بنية مخصصة](./create_a_model) لمزيد من المعلومات حول تكوينات مخصصة.
بالتأكيد! فيما يلي النص المترجم مع مراعاة التعليمات التي قدمتها:

## المدرب - حلقة تدريبية مُحُلّلة لـ PyTorch
يمكن استخدام جميع النماذج كـ [`torch.nn.Module`] قياسي، لذا يمكنك استخدامها في أي حلقة تدريبية نموذجية. في حين يمكنك كتابة حلقة التدريب الخاصة بك، يوفر 🤗 Transformers فئة [`Trainer`] لـ PyTorch، والتي تحتوي على حلقة التدريب الأساسية وتضيف وظائف إضافية لميزات مثل التدريب الموزع والدقة المختلطة، وغير ذلك الكثير.

اعتمادًا على مهمتك، عادةً ما تقوم بتمرير المعلمات التالية إلى [`Trainer`]:

1. ستبدأ بـ [`PreTrainedModel`] أو [`torch.nn.Module`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

2. تحتوي [`TrainingArguments`] على فرط معلمات النموذج التي يمكنك تغييرها مثل معدل التعلم وحجم الدفعة وعدد العصور التي يجب التدريب عليها. يتم استخدام القيم الافتراضية إذا لم تحدد أي حجج تدريب:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(
...     output_dir="path/to/save/folder/",
...     learning_rate=2e-5,
...     per_device_train_batch_size=8,
...     per_device_eval_batch_size=8,
...     num_train_epochs=2,
... )
```

3. قم بتحميل فئة ما قبل المعالجة مثل tokenizer، أو معالج الصور، أو مستخرج الميزات، أو المعالج:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

4. تحميل مجموعة بيانات:

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("rotten_tomatoes")
```

5. قم بإنشاء دالة لتحويل مجموعة البيانات إلى رموز:

```py
>>> def tokenize_dataset(dataset):
...     return tokenizer(dataset["text"])
```

ثم قم بتطبيقه على مجموعة البيانات بأكملها مع [`~ datasets.Dataset.map`]:

```py
>>> dataset = dataset.map(tokenize_dataset, batched=True)
```

6. [`DataCollatorWithPadding`] لإنشاء دفعة من الأمثلة من مجموعة البيانات الخاصة بك:

```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

الآن قم بتجميع جميع هذه الفئات في [`Trainer`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )
```

عندما تكون مستعدًا، اتصل بـ [`~ Trainer.train`] لبدء التدريب:

```py
>>> trainer.train()
```

<Tip>
بالنسبة للمهام - مثل الترجمة أو التلخيص - التي تستخدم نموذج تسلسل إلى تسلسل، استخدم فئات [`Seq2SeqTrainer`] و [`Seq2SeqTrainingArguments`] بدلاً من ذلك.
</Tip>

يمكنك تخصيص سلوك حلقة التدريب عن طريق إنشاء فئة فرعية من الطرق داخل [`Trainer`]. يسمح لك ذلك بتخصيص ميزات مثل دالة الخسارة والمُحَمِّل والجدول الزمني. الق نظرة على المرجع [`Trainer`] للطرق التي يمكن إنشاء فئات فرعية منها.

والطريقة الأخرى لتخصيص حلقة التدريب هي باستخدام [المستدعيات]. يمكنك استخدام المستدعيات للتكامل مع مكتبات أخرى وفحص حلقة التدريب للإبلاغ عن التقدم المحرز أو إيقاف التدريب مبكرًا. لا تعدل المستدعيات أي شيء في حلقة التدريب نفسها. لتخصيص شيء مثل دالة الخسارة، تحتاج إلى إنشاء فئة فرعية من [`Trainer`] بدلاً من ذلك.

## تدريب مع TensorFlow
جميع النماذج هي [`tf.keras.Model`] قياسي، لذا يمكن تدريبها في TensorFlow باستخدام واجهة برمجة تطبيقات Keras. يوفر 🤗 Transformers طريقة [`~ TFPreTrainedModel.prepare_tf_dataset`] لتحميل مجموعة البيانات الخاصة بك بسهولة كـ `tf.data.Dataset` حتى تتمكن من البدء في التدريب على الفور باستخدام طرق `compile` و`fit` من Keras.

1. ستبدأ بـ [`TFPreTrainedModel`] أو [`tf.keras.Model`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

2. قم بتحميل فئة ما قبل المعالجة مثل tokenizer، أو معالج الصور، أو مستخرج الميزات، أو المعالج:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

3. قم بإنشاء دالة لتحويل مجموعة البيانات إلى رموز:

```py
>>> def tokenize_dataset(dataset):
...     return tokenizer(dataset["text"])
```

4. قم بتطبيق tokenizer على مجموعة البيانات بأكملها مع [`~ datasets.Dataset.map`] ثم قم بتمرير مجموعة البيانات و tokenizer إلى [`~ TFPreTrainedModel.prepare_tf_dataset`]. يمكنك أيضًا تغيير حجم الدفعة وخلط مجموعة البيانات هنا إذا أردت:

```py
>>> dataset = dataset.map(tokenize_dataset)
>>> tf_dataset = model.prepare_tf_dataset(
...     dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
... )
```

5. عندما تكون مستعدًا، يمكنك استدعاء `compile` و`fit` لبدء التدريب. لاحظ أن نماذج Transformers تحتوي جميعها على دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذا فأنت لست بحاجة إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> from tensorflow.keras.optimizers import Adam

>>> model.compile(optimizer='adam') # لا توجد حجة الخسارة!
>>> model.fit(tf_dataset)
```

## ماذا بعد؟
الآن بعد أن أكملت الجولة السريعة لـ 🤗 Transformers، اطلع على أدلةنا وتعرف على كيفية القيام بأشياء أكثر تحديدًا مثل كتابة نموذج مخصص، وتنقيح نموذج لمهمة، وكيفية تدريب نموذج باستخدام نص برمجي. إذا كنت مهتمًا بمعرفة المزيد عن المفاهيم الأساسية لـ 🤗 Transformers، فاحصل على فنجان من القهوة وخذ نظرة على أدلةنا المفاهيمية!