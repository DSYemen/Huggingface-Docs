## الترجمة

تعد الترجمة إحدى المهام التي يمكن صياغتها كمشكلة تسلسل إلى تسلسل، وهو إطار عمل قوي لإرجاع بعض المخرجات من الإدخال، مثل الترجمة أو الملخص. وتُستخدم أنظمة الترجمة عادة للترجمة بين نصوص اللغات المختلفة، ولكن يمكن أيضًا استخدامها للكلام أو بعض المزيج بينهما مثل النص إلى كلام أو الكلام إلى نص.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج T5 الدقيق على الجزء الفرعي الإنجليزي الفرنسي من مجموعة بيانات OPUS Books لترجمة النص الإنجليزي إلى الفرنسية.
2. استخدام النموذج الدقيق للاستنتاج.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate sacrebleu
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات OPUS Books

ابدأ بتحميل الجزء الفرعي الإنجليزي الفرنسي من مجموعة بيانات [OPUS Books](https://huggingface.co/datasets/opus_books) من مكتبة Datasets:

```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

قسِّم مجموعة البيانات إلى مجموعتين للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:

```py
>>> books["train"][0]
{'id': '90560',
'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`: ترجمة إنجليزية وفرنسية للنص.

## معالجة مسبقة

الخطوة التالية هي تحميل برنامج تشفير T5 لمعالجة أزواج اللغة الإنجليزية والفرنسية:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

يجب أن تقوم دالة المعالجة المسبقة التي تريد إنشائها بما يلي:

1. إضافة بادئة إلى الإدخال باستخدام موجه حتى يعرف T5 أن هذه مهمة ترجمة. تتطلب بعض النماذج القادرة على مهام NLP متعددة موجهات لمهام محددة.
2. تشفير الإدخال (الإنجليزية) والهدف (الفرنسية) بشكل منفصل لأنه لا يمكن تشفير النص الفرنسي باستخدام برنامج تشفير تم تدريبه مسبقًا على مفردات اللغة الإنجليزية.
3. اقطع التسلسلات بحيث لا تكون أطول من الطول الأقصى المحدد بواسطة معلمة `max_length`.

```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "


>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] في مكتبة Datasets. يمكنك تسريع وظيفة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForSeq2Seq`]. من الأكثر كفاءة *حشو* الجمل ديناميكيًا إلى الطول الأطول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>

<pt>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

</pt>

<tf>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```

</tf>

</frameworkcontent>

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل مقياس [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) (راجع جولة التقييم السريعة في Hugging Face لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب درجة SacreBLEU:

```py
>>> import numpy as np


>>> def postprocess_text(preds، labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels


>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```

دالة `compute_metrics` الخاصة بك جاهزة الآن، وستعود إليها عند إعداد التدريب.
## التدريب

إذا لم تكن معتاداً على ضبط نموذج باستخدام [`Trainer`] ، فراجع البرنامج التعليمي الأساسي [هنا] (../training # train-with-pytorch-trainer) !

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل T5 مع [`AutoModelForSeq2SeqLM`] :

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`Seq2SeqTrainingArguments`] . المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. سوف تقوم بإرسال هذا النموذج إلى المحور عن طريق تعيين `push_to_hub=True` (يجب أن تكون قد سجلت الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم مقياس SacreBLEU وحفظ نقطة تفتيش التدريب.

2. قم بتمرير الحجج التدريبية إلى [`Seq2SeqTrainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمحلل اللغوي ومجمع البيانات و `compute_metrics` function.

3. استدعاء [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model"،
...     eval_strategy="epoch"،
...     learning_rate=2e-5،
...     per_device_train_batch_size=16،
...     per_device_eval_batch_size=16،
...     weight_decay=0.01،
...     save_total_limit=3،
...     num_train_epochs=2،
...     predict_with_generate=True،
...     fp16=True،
...     push_to_hub=True،
... )

>>> trainer = Seq2SeqTrainer(
...     model=model،
...     args=training_args،
...     train_dataset=tokenized_books ["train"]،
...     eval_dataset=tokenized_books ["test"]،
...     tokenizer=tokenizer،
...     data_collator=data_collator،
...     compute_metrics=compute_metrics،
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك في المحور باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا] (../training # train-a-tensorflow-model-with-keras) !

لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن ومعدل تعلم وجدول، وبعض فرط معلمات التدريب:

```py
>>> from transformers import AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

بعد ذلك، يمكنك تحميل T5 مع [`TFAutoModelForSeq2SeqLM`] :

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`] :

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_books ["train"]،
...     shuffle=True،
...     batch_size=16،
...     collate_fn=data_collator،
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_books ["test"]،
...     shuffle=False،
...     batch_size=16،
...     collate_fn=data_collator،
... )
```

قم بتكوين النموذج للتدريب مع [`compile`] (https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer) # لا يوجد حجة الخسارة!
```

الأمران الأخيران اللذان يجب إعدادهما قبل بدء التدريب هما حساب مقياس SacreBLEU من التوقعات، وتوفير طريقة لدفع نموذجك إلى المحور. يتم تنفيذ كلاهما باستخدام [Keras callbacks] (../main_classes/keras_callbacks).

مرر دالتك `compute_metrics` إلى [`~transformers.KerasMetricCallback`] :

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics، eval_dataset=tf_validation_set)
```

حدد مكان دفع نموذجك والمحلل اللغوي في [`~transformers.PushToHubCallback`] :

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_opus_books_model"،
...     tokenizer=tokenizer،
... )
```

بعد ذلك، قم بتجميع مكالماتك مرة أخرى:

```py
>>> callbacks = [metric_callback، push_to_hub_callback]
```

أخيرًا، أنت على استعداد لبدء تدريب نموذجك! اتصل [`fit`] (https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، ومكالماتك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set، validation_data=tf_test_set، epochs=3، callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى المحور حتى يتمكن الجميع من استخدامه!

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج للترجمة، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)
أو [دفتر TensorFlow] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb).

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

فكر في بعض النصوص التي ترغب في ترجمتها إلى لغة أخرى. بالنسبة إلى T5، يجب أن تقوم بتحديد بادئة إدخالك وفقًا للمهمة التي تعمل عليها. للترجمة من الإنجليزية إلى الفرنسية، يجب أن تقوم بتحديد بادئة إدخالك كما هو موضح أدناه:

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`] . قم بتنفيذ مثيل `pipeline` للترجمة باستخدام نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

# تغيير `xx` إلى لغة الإدخال و `yy` إلى لغة الإخراج المطلوبة.
# أمثلة: "en" للإنجليزية، "fr" للفرنسية، "de" للألمانية، "es" للإسبانية، "zh" للصينية، إلخ؛ translation_en_to_fr يترجم الإنجليزية إلى الفرنسية
# يمكنك عرض جميع قوائم اللغات هنا - https://huggingface.co/languages
>>> translator = pipeline ("translation_xx_to_yy"، model="my_awesome_opus_books_model")
>>> translator (text)
[{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.}]
```

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا كنت تريد ذلك:

قم بتوكينز النص وإرجاع `input_ids` كتوترات PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained ("my_awesome_opus_books_model")
>>> المدخلات = tokenizer (النص، return_tensors="pt"). input_ids
```

استخدم طريقة [`~generation.GenerationMixin.generate`] لإنشاء الترجمة. لمزيد من التفاصيل حول استراتيجيات إنشاء النصوص المختلفة والمعلمات للتحكم في الإنشاء، راجع [Text Generation] (../main_classes/text_generation) API.

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained ("my_awesome_opus_books_model")
>>> outputs = model.generate (inputs، max_new_tokens=40، do_sample=True، top_k=30، top_p=0.95)
```

فك تشفير معرفات الرموز المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode (outputs [0]، skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```

قم بتوكينز النص وإرجاع `input_ids` كتوترات TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained ("my_awesome_opus_books_model")
>>> المدخلات = tokenizer (النص، return_tensors="tf"). input_ids
```

استخدم طريقة [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] لإنشاء الترجمة. لمزيد من التفاصيل حول استراتيجيات إنشاء النصوص المختلفة والمعلمات للتحكم في الإنشاء، راجع [Text Generation] (../main_classes/text_generation) API.

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained ("my_awesome_opus_books_model")
>>> outputs = model.generate (inputs، max_new_tokens=40، do_sample=True، top_k=30، top_p=0.95)
```

فك تشفير معرفات الرموز المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode (outputs [0]، skip_special_tokens=True)
'Les lugumes partagent les ressources avec des bactéries fixatrices d'azote.'
```