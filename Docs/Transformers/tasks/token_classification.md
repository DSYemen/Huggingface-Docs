## تصنيف الرموز 

يُعيّن تصنيف الرموز علامة تصنيف لكل رمز في جملة. إحدى مهام تصنيف الرموز الشائعة هي التعرف على الكيانات المسماة (NER). تحاول NER إيجاد تسمية لكل كيان في جملة، مثل شخص أو موقع أو منظمة.

سيوضح لك هذا الدليل كيفية:

1. ضبط دقة [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) على مجموعة بيانات [WNUT 17](https://huggingface.co/datasets/wnut_17) للكشف عن كيانات جديدة.
2. استخدام النموذج المضبوط دقة الخاص بك للاستنتاج.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate seqeval
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات WNUT 17 

ابدأ بتحميل مجموعة بيانات WNUT 17 من مكتبة Datasets 🤗:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

ثم الق نظرة على مثال:

```py
>>> wnut["train"][0]
{'id': '0',
'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

يمثل كل رقم في `ner_tags` كيانًا. قم بتحويل الأرقام إلى أسماء تسمياتها لمعرفة ماهية الكيانات:

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
"O",
"B-corporation",
"I-corporation",
"B-creative-work",
"I-creative-work",
"B-group",
"I-group",
"B-location",
"I-location",
"B-person",
"I-person",
"B-product",
"I-product",
]
```

يشير الحرف الذي يسبق كل `ner_tag` إلى موضع الرمز الخاص بالكيان:

- `B-` يشير إلى بداية الكيان.
- `I-` يشير إلى أن الرمز موجود داخل نفس الكيان (على سبيل المثال، رمز `State` هو جزء من كيان مثل `Empire State Building`).
- `0` يشير إلى أن الرمز لا يقابل أي كيان.

## معالجة مسبقة 

الخطوة التالية هي تحميل معالج رموز DistilBERT لمعالجة حقل `tokens`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

كما رأيت في حقل `tokens` المثال أعلاه، يبدو أن المدخلات قد تم تقسيمها إلى رموز بالفعل. لكن المدخلات لم يتم تقسيمها إلى رموز بالفعل، وستحتاج إلى تعيين `is_split_into_words=True` لتقسيم الكلمات إلى رموز فرعية. على سبيل المثال:

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

ومع ذلك، فإن هذا يضيف بعض الرموز الخاصة `[CLS]` و`[SEP]`، وتؤدي تقسيم الكلمات إلى رموز فرعية إلى عدم تطابق بين المدخلات والتسميات. قد تنقسم الآن كلمة واحدة تقابل تسمية واحدة إلى رمزين فرعيين. ستحتاج إلى إعادة محاذاة الرموز والعلامات من خلال:

1. قم بتعيين جميع الرموز إلى كلماتها المقابلة باستخدام طريقة [`word_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.word_ids).
2. قم بتعيين التسمية `-100` إلى الرموز الخاصة `[CLS]` و`[SEP]` حتى يتم تجاهلها بواسطة دالة الخسارة PyTorch (انظر [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
3. قم بتسمية الرمز الأول للكلمة فقط. قم بتعيين `-100` إلى الرموز الفرعية الأخرى من نفس الكلمة.

هنا كيف يمكنك إنشاء وظيفة لإعادة محاذاة الرموز والعلامات، وقص التسلسلات بحيث لا تكون أطول من طول الإدخال الأقصى لـ DistilBERT:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # قم بتعيين الرموز إلى كلماتها المقابلة.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # قم بتعيين التسميات الخاصة إلى -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # قم بتسمية الرمز الأول للكلمة فقط.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

لتطبيق وظيفة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`] في مكتبة Datasets 🤗. يمكنك تسريع وظيفة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorWithPadding`]. من الأكثر كفاءة *التقسيم الديناميكي* للجمل إلى أطول طول في دفعة أثناء الجمع، بدلاً من تقسيم مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>

<pt>

```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

</pt>

<tf>

```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
```

</tf>

</frameworkcontent>

## تقييم 

غالبًا ما يكون من المفيد تضمين مقياس أثناء التدريب لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل إطار عمل [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس). في الواقع، ينتج Seqeval عدة درجات: الدقة والاستدعاء وF1 والدقة.

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

احصل على تسميات NER أولاً، ثم قم بإنشاء وظيفة تمرر تنبؤاتك الصحيحة وتسمياتك الصحيحة إلى [`~evaluate.EvaluationModule.compute`] لحساب الدرجات:

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

وظيفة `compute_metrics` الخاصة بك جاهزة الآن، وستعود إليها عند إعداد التدريب الخاص بك.
## التدريب

قبل البدء في تدريب النموذج الخاص بك، قم بإنشاء خريطة من المعرفات المتوقعة إلى تسمياتها باستخدام `id2label` و `label2id`:

```python
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<frameworkcontent>
<pt>
<Tip>
إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`]، فراجع البرنامج التعليمي الأساسي [هنا] (../training # train-with-pytorch-trainer)
</Tip>

أنت الآن على استعداد لبدء تدريب النموذج الخاص بك! قم بتحميل DistilBERT مع [`AutoModelForTokenClassification`] جنبًا إلى جنب مع عدد التسميات المتوقعة، وخرائط التسميات:

```python
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد أين يتم حفظ نموذجك. ستقوم بالدفع بهذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم درجات seqeval وحفظ نقطة تفتيش التدريب.

2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمحلل اللغوي ومجمع البيانات ووظيفة `compute_metrics`.

3. استدعاء [`~Trainer.train`] لضبط نموذجك.

```python
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model"،
...     learning_rate=2e-5،
...     per_device_train_batch_size=16،
...     per_device_eval_batch_size=16،
...     num_train_epochs=2،
...     weight_decay=0.01،
...     eval_strategy="epoch"،
...     save_strategy="epoch"،
...     load_best_model_at_end=True،
...     push_to_hub=True،
... )

>>> trainer = Trainer(
...     model=model،
...     args=training_args،
...     train_dataset=tokenized_wnut["train"]،
...     eval_dataset=tokenized_wnut["test"]،
...     tokenizer=tokenizer،
...     data_collator=data_collator،
...     compute_metrics=compute_metrics،
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك مع Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```python
>>> trainer.push_to_hub()
```

</pt>
<tf>
<Tip>
إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا] (../training # train-a-tensorflow-model-with-keras)
</Tip>

لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن، وجدول معدل التعلم، وبعض فرط معلمات التدريب:

```python
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 3
>>> num_train_steps = (len (tokenized_wnut ["train"]) // batch_size) * num_train_epochs
>>> optimizer، lr_schedule = create_optimizer (
...     init_lr=2e-5،
...     num_train_steps=num_train_steps،
...     weight_decay_rate=0.01،
...     num_warmup_steps=0،
... )
```

بعد ذلك، يمكنك تحميل DistilBERT مع [`TFAutoModelForTokenClassification`] جنبًا إلى جنب مع عدد التسميات المتوقعة، وخرائط التسميات:

```python
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased"، num_labels=13، id2label=id2label، label2id=label2id
... )
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```python
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_wnut ["train"]،
...     shuffle=True،
...     batch_size=16،
...     collate_fn=data_collator،
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_wnut ["validation"]،
...     shuffle=False،
...     batch_size=16،
...     collate_fn=data_collator،
... )
```

قم بتكوين النموذج للتدريب باستخدام [`compile`] (https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers بها دالة خسارة افتراضية ذات صلة بالمهمة، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```python
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer) # لا توجد حجة الخسارة!
```

الأمران الأخيران اللذان يجب إعدادهما قبل بدء التدريب هما حساب درجات seqeval من التوقعات، وتوفير طريقة لدفع نموذجك إلى Hub. يتم تنفيذ كلاهما باستخدام [Keras callbacks] (../main_classes/keras_callbacks).

مرر دالتك `compute_metrics` إلى [`~transformers.KerasMetricCallback`]:

```python
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics، eval_dataset=tf_validation_set)
```

حدد أين تدفع نموذجك ومحلل اللغة الخاص بك في [`~transformers.PushToHubCallback`]:

```python
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_wnut_model"،
...     tokenizer=tokenizer،
... )
```

بعد ذلك، قم بتجميع مكالماتك مرة أخرى:

```python
>>> callbacks = [metric_callback، push_to_hub_callback]
```

أخيرًا، أنت على استعداد لبدء تدريب نموذجك! استدعاء [`fit`] (https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، ومكالماتك لضبط نموذجك:

```python
>>> model.fit(x=tf_train_set، validation_data=tf_validation_set، epochs=3، callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!

</tf>
</frameworkcontent>

<Tip>
لمثال أكثر عمقًا حول كيفية ضبط نموذج للتصنيف الرمزي، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)
أو [دفتر ملاحظات TensorFlow] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
</Tip>

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

احصل على بعض النصوص التي تريد تشغيل الاستدلال عليها:

```python
>>> text = "Golden State Warriors هي فريق كرة سلة أمريكي محترف يقع مقره في سان فرانسيسكو."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه في [`pipeline`]. قم بتنفيذ مثيل `pipeline` لـ NER مع نموذجك، ومرر نصك إليه:

```python
>>> from transformers import pipeline

>>> classifier = pipeline("ner"، model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location'،
'score': 0.42658573،
'index': 2،
'word': 'golden'،
'start': 4،
'end': 10}،
{'entity': 'I-location'،
'score': 0.35856336،
'index': 3،
'word': 'state'،
'start': 11،
'end': 16}،
{'entity': 'B-group'،
'score': 0.3064001،
'index': 4،
'word': 'warriors'،
'start': 17،
'end': 25}،
{'entity': 'B-location'،
'score': 0.65523505،
'index': 13،
'word': 'san'،
'start': 80،
'end': 83}،
{'entity': 'B-location'،
'score': 0.4668663،
'index': 14،
'word': 'francisco'،
'start': 84،
'end': 93}]
```

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا أردت:

<frameworkcontent>
<pt>
قم بتحليل النص وإرجاع tensers PyTorch:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text، return_tensors="pt")
```

مرر المدخلات إلى النموذج وإرجاع `logits`:

```python
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model (** inputs). logits
```

احصل على الفئة ذات أعلى احتمال، واستخدم تعيين `id2label` للنموذج لتحويله إلى تسمية نصية:

```python
>>> predictions = torch.argmax (logits، dim=2)
>>> predicted_token_class = [model.config.id2label [t.item()] for t in predictions [0]]
>>> predicted_token_class
['O'،
'O'،
'B-location'،
'I-location'،
'B-group'،
'O'،
'O'،
'O'،
'O'،
'O'،
'O'،
'O'،
'O'،
'B-location'،
'B-location'،
'O'،
'O']
```

</pt>
<tf>
قم بتحليل النص وإرجاع tensers TensorFlow:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text، return_tensors="tf")
```

مرر المدخلات إلى النموذج وإرجاع `logits`:

```python
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> logits = model (** inputs). logits
```

احصل على الفئة ذات أعلى احتمال، واستخدم تعيين `id2label` للنموذج لتحويله إلى تسمية نصية:

```python
>>> predicted_token_class_ids = tf.math.argmax (logits، axis=-1)
>>> predicted_token_class = [model.config.id2label [t] for t in predicted_token_class_ids [0]. numpy(). tolist()]
>>> predicted_token_class
['O'،
'O'،
'B-location'،
'I-location'،
'B-group'،
'O'،
'O'،
'O'،
'O'،
'O'،
'O'،
'O'،
'O'،
'B-location'،
'B-location'،
'O'،
'O']
```

</tf>
</frameworkcontent>