## الاختيار من متعدد

تتشابه مهمة الاختيار من متعدد مع مهمة الإجابة على الأسئلة، باستثناء أنه يتم توفير عدة إجابات مرشحة إلى جانب السياق، ويتم تدريب النموذج لاختيار الإجابة الصحيحة.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج BERT الدقيق على التهيئة "العادية" لمجموعة بيانات SWAG لاختيار أفضل إجابة من بين عدة خيارات وسياق معين.
2. استخدام النموذج المضبوط دقيقًا للاستنتاج.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات SWAG

ابدأ بتحميل التهيئة "العادية" لمجموعة بيانات SWAG من مكتبة Datasets 🤗:

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

ثم الق نظرة على مثال:

```py
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
'ending1': 'has heard approaching them.',
'ending2': "arrives and they're outside dancing and asleep.",
'ending3': 'turns the lead singer watches the performance.',
'fold-ind': '3416',
'gold-source': 'gold',
'label': 0,
'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
'sent2': 'A drum line',
'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
'video-id': 'anetv_jkn6uvmqwh4'}
```

على الرغم من أنه يبدو أن هناك العديد من الحقول هنا، إلا أنها في الواقع واضحة ومباشرة:

- `sent1` و `sent2`: توضح هذه الحقول كيف تبدأ الجملة، وإذا قمت بدمج الاثنين معًا، فستحصل على حقل `startphrase`.
- `ending`: يقترح نهاية محتملة لكيفية انتهاء الجملة، ولكن واحدة منها فقط صحيحة.
- `label`: يحدد نهاية الجملة الصحيحة.

## معالجة مسبقة

الخطوة التالية هي تحميل محدد رموز BERT لمعالجة بدايات الجمل والنهايات الأربع المحتملة:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

تحتاج دالة المعالجة المسبقة التي تريد إنشاؤها إلى:

1. عمل أربع نسخ من حقل `sent1` ودمج كل منها مع `sent2` لإعادة إنشاء كيفية بدء الجملة.
2. دمج `sent2` مع كل من النهايات الأربع المحتملة للجملة.
3. تسطيح هاتين القائمتين حتى تتمكن من توكيدهما، ثم إلغاء تسطيحهما لاحقًا بحيث يكون لكل مثال حقول `input_ids` و `attention_mask` و `labels` المقابلة.

```py
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] في مكتبة 🤗 Datasets. يمكنك تسريع دالة `map` عن طريق تعيين `batched=True` لمعالجة عدة عناصر من مجموعة البيانات في وقت واحد:

```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

لا تحتوي مكتبة 🤗 Transformers على جامع بيانات للاختيار من متعدد، لذلك ستحتاج إلى تكييف جامع البيانات [`DataCollatorWithPadding`] لإنشاء دفعة من الأمثلة. من الأكثر كفاءة أن تقوم بالتحويل الديناميكي للمجموعات إلى الطول الأطول في دفعة أثناء التجميع، بدلاً من تحويل مجموعة البيانات بأكملها إلى الطول الأقصى.

يقوم `DataCollatorForMultipleChoice` بتسطيح جميع مدخلات النموذج، ثم تطبيق التحويل، ثم إلغاء تسطيح النتائج:

<frameworkcontent>
<pt>
```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import torch


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="pt",
...         )

...         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
...         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
...         return batch
```
</pt>
<tf>
```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import tensorflow as tf


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="tf",
...         )

...         batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
...         batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
...         return batch
```
</tf>
</frameworkcontent>

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب المقياس):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

دالتك `compute_metrics` جاهزة الآن، وستعود إليها عند إعداد تدريبك.
## التدريب

أنت الآن مستعد لبدء تدريب نموذجك! قم بتحميل BERT باستخدام [AutoModelForMultipleChoice]:

```python
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [TrainingArguments]. المعلمة المطلوبة الوحيدة هي output_dir التي تحدد مكان حفظ نموذجك. سوف تقوم بدفع هذا النموذج إلى المحور عن طريق تعيين push_to_hub=True (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [Trainer] بتقييم الدقة وحفظ نقطة تفتيش التدريب.

2. قم بتمرير فرط معلمات التدريب إلى [Trainer] جنبًا إلى جنب مع النموذج ومجموعة البيانات ومعيّن الرموز وملف تجميع البيانات ووظيفة compute_metrics.

3. استدعاء [~ Trainer.train] لضبط نموذجك بشكل دقيق.

```python
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model"،
...     eval_strategy="epoch"،
...     save_strategy="epoch"،
...     load_best_model_at_end=True،
...     learning_rate=5e-5،
...     per_device_train_batch_size=16،
...     per_device_eval_batch_size=16،
...     num_train_epochs=3،
...     weight_decay=0.01،
...     push_to_hub=True،
... )

>>> trainer = Trainer(
...     model=model،
...     args=training_args،
...     train_dataset=tokenized_swag["train"]،
...     eval_dataset=tokenized_swag["validation"]،
...     tokenizer=tokenizer،
...     data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer)،
...     compute_metrics=compute_metrics،
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك في المحور باستخدام طريقة [~ transformers.Trainer.push_to_hub] حتى يتمكن الجميع من استخدام نموذجك:

```python
>>> trainer.push_to_hub()
```

لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن ومعدل تعلم وجدول زمني وبعض فرط معلمات التدريب:

```python
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

ثم يمكنك تحميل BERT باستخدام [TFAutoModelForMultipleChoice]:

```python
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [~ transformers.TFPreTrainedModel.prepare_tf_dataset]:

```python
>>> data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_swag["train"]،
...     shuffle=True،
...     batch_size=batch_size،
...     collate_fn=data_collator،
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_swag["validation"]،
...     shuffle=False،
...     batch_size=batch_size،
...     collate_fn=data_collator،
... )
```

قم بتكوين النموذج للتدريب باستخدام [compile](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers بها دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```python
>>> model.compile(optimizer=optimizer)  # لا توجد وسيطة خسارة!
```

الأمران الأخيران اللذان يجب إعدادهما قبل بدء التدريب هما حساب الدقة من التوقعات، وتوفير طريقة لدفع نموذجك إلى المحور. يتم تنفيذ كلاهما باستخدام [Keras callbacks](../main_classes/keras_callbacks).

مرر دالتك `compute_metrics` إلى [~ transformers.KerasMetricCallback]:

```python
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد مكان دفع نموذجك ومعيّن الرموز في [~ transformers.PushToHubCallback]:

```python
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model"،
...     tokenizer=tokenizer،
... )
```

ثم قم بتجميع استدعاءاتك مرة أخرى:

```python
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! استدعاء [fit](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، واستدعاءاتك لضبط النموذج بدقة:

```python
>>> model.fit(x=tf_train_set، validation_data=tf_validation_set، epochs=2، callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى المحور حتى يتمكن الجميع من استخدامه!

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج للاختيار من متعدد، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
أو [دفتر TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

فكر في بعض النصوص وإجابتين مرشحتين:

```python
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

قم بمعالجة كل زوج من النصوص والإجابات المرشحة وقم بإرجاع tensers PyTorch. يجب عليك أيضًا إنشاء بعض `labels`:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]]، return_tensors="pt"، padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

مرر المدخلات والملصقات الخاصة بك إلى النموذج وأعد الوسيطات:

```python
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

احصل على الفئة ذات الاحتمالية الأعلى:

```python
>>> predicted_class = logits.argmax().item()
>>> predicted_class
'0'
```

قم بمعالجة كل زوج من النصوص والإجابات المرشحة وإرجاع tensers TensorFlow:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]]، return_tensors="tf"، padding=True)
```

مرر المدخلات إلى النموذج وأعد الوسيطات:

```python
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

احصل على الفئة ذات الاحتمالية الأعلى:

```python
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
'0'
```