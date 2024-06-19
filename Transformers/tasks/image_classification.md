## تصنيف الصور

يُخصص تصنيف الصور علامة أو فئة لصورة معينة. وعلى عكس تصنيف النصوص أو الصوتيات، فإن المدخلات هي قيم البكسل التي تتكون منها الصورة. هناك العديد من التطبيقات لتصنيف الصور، مثل الكشف عن الأضرار بعد وقوع كارثة طبيعية، أو مراقبة صحة المحاصيل، أو المساعدة في فحص الصور الطبية للكشف عن علامات المرض.

يوضح هذا الدليل كيفية:

1. ضبط نموذج [ViT](model_doc/vit) بدقة على مجموعة بيانات [Food-101](https://huggingface.co/datasets/food101) لتصنيف عنصر غذائي في صورة.
2. استخدام النموذج المضبوط بدقة للاستنتاج.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات Food-101

ابدأ بتحميل جزء فرعي أصغر من مجموعة بيانات Food-101 من مكتبة 🤗 Datasets. سيعطيك هذا فرصة لتجربة النموذج والتأكد من أن كل شيء يعمل بشكل صحيح قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

قسِّم مجموعة البيانات إلى مجموعات فرعية للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> food = food.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
'label': 79}
```

يحتوي كل مثال في مجموعة البيانات على حقلين:

- `image`: صورة PIL لعنصر الطعام
- `label`: فئة التصنيف لعنصر الطعام

لجعل الأمر أسهل بالنسبة للنموذج للحصول على اسم التصنيف من معرف التصنيف، قم بإنشاء قاموس يحدد أسماء التصنيفات إلى قيمة عددية والعكس:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

الآن يمكنك تحويل معرف التصنيف إلى اسم التصنيف:

```py
>>> id2label[str(79)]
'prime_rib'
```

## معالجة مسبقة

الخطوة التالية هي تحميل معالج صور ViT لمعالجة الصورة وتحويلها إلى مصفوفة:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

<frameworkcontent>
<pt>
قم بتطبيق بعض التحويلات على الصور لجعل النموذج أكثر قوة ضد الإفراط في التخصيص. هنا، ستستخدم وحدة [`transforms`](https://pytorch.org/vision/stable/transforms.html) من مكتبة torchvision، ولكن يمكنك أيضًا استخدام أي مكتبة صور تفضلها.

اقتص جزءًا عشوائيًا من الصورة، وقم بتغيير حجمها، وطبِّعها باستخدام المتوسط والانحراف المعياري للصورة:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

ثم قم بإنشاء دالة معالجة مسبقة لتطبيق التحويلات وإرجاع `pixel_values` - المدخلات إلى النموذج - للصورة:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.with_transform`] من مكتبة 🤗 Datasets. يتم تطبيق التحويلات أثناء التنقل عند تحميل عنصر من مجموعة البيانات:

```py
>>> food = food.with_transform(transforms)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DefaultDataCollator`]. على عكس أدوات الجمع الأخرى في مكتبة 🤗 Transformers، فإن `DefaultDataCollator` لا تطبق معالجة مسبقة إضافية مثل التوسيد.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
لتجنب الإفراط في التخصيص وجعل النموذج أكثر قوة، أضف بعض التعزيزات للبيانات إلى الجزء التدريبي من مجموعة البيانات.

هنا، نستخدم طبقات ما قبل المعالجة من Keras لتحديد التحويلات لبيانات التدريب (بما في ذلك التعزيزات)، والتحويلات لبيانات التحقق (الاقتصاص المركزي فقط، تغيير الحجم، والتوحيد). يمكنك استخدام `tf.image` أو أي مكتبة أخرى تفضلها.

```py
>>> from tensorflow import keras
>>> from tensorflow.keras import layers

>>> size = (image_processor.size["height"], image_processor.size["width"])

>>> train_data_augmentation = keras.Sequential(
...     [
...         layers.RandomCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...         layersMultiplier
...         layers.RandomFlip("horizontal"),
...         layers.RandomRotation(factor=0.02),
...         layers.RandomZoom(height_factor=0.2, width_factor=0.2),
...     ],
...     name="train_data_augmentation",
... )

>>> val_data_augmentation = keras.Sequential(
...     [
...         layers.CenterCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...     ],
...     name="val_data_augmentation",
... )
```

بعد ذلك، قم بإنشاء دوال لتطبيق التحويلات المناسبة على دفعة من الصور، بدلاً من صورة واحدة في كل مرة.

```py
>>> import numpy as np
>>> import tensorflow as tf
>>> from PIL import Image


>>> def convert_to_tf_tensor(image: Image):
...     np_image = np.array(image)
...     tf_image = tf.convert_to_tensor(np_image)
...     # `expand_dims()` is used to add a batch dimension since
...     # the TF augmentation layers operates on batched inputs.
...     return tf.expand_dims(tf_image, 0)


>>> def preprocess_train(example_batch):
...     """Apply train_transforms across a batch."""
...     images = [
...         train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch


... def preprocess_val(example_batch):
...     """Apply val_transforms across a batch."""
...     images = [
...         val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch
```

استخدم طريقة [`~datasets.Dataset.set_transform`] من مكتبة 🤗 Datasets لتطبيق التحويلات أثناء التنقل:

```py
food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)
```

وكخطوة معالجة مسبقة نهائية، قم بإنشاء دفعة من الأمثلة باستخدام `DefaultDataCollator`. على عكس أدوات الجمع الأخرى في مكتبة 🤗 Transformers، فإن `DefaultDataCollator` لا تطبق معالجة مسبقة إضافية، مثل التوسيد.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```

</tf>
</frameworkcontent>

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع الدليل السريع لـ 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتصنيفاتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

الآن أصبحت دالة `compute_metrics` الخاصة بك جاهزة للعمل، وستعود إليها عند إعداد التدريب.
بالتأكيد! فيما يلي ترجمة للجزء المطلوب من النص إلى اللغة العربية:

## التدريب

إذا لم تكن معتاداً على ضبط نموذج باستخدام [`Trainer`]، فراجع الدليل التعليمي الأساسي [هنا] (../training#train-with-pytorch-trainer)

الآن أنت مستعد لبدء تدريب نموذجك! قم بتحميل ViT مع [`AutoModelForImageClassification`]. حدد عدد التصنيفات جنبًا إلى جنب مع عدد التصنيفات المتوقعة، وتخطيط التصنيفات:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. من المهم ألا تقوم بإزالة الأعمدة غير المستخدمة لأن ذلك سيؤدي إلى إسقاط عمود "الصورة". بدون عمود "الصورة"، لا يمكنك إنشاء "pixel_values". قم بتعيين `remove_unused_columns=False` لمنع هذا السلوك! المعلمة المطلوبة الوحيدة الأخرى هي `output_dir` التي تحدد مكان حفظ نموذجك. سوف تقوم بإرسال هذا النموذج إلى المحور عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم الدقة وحفظ نقطة تفتيش التدريب.

2. مرر حُجج التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات ومعالج الرموز ومجمع البيانات ووظيفة `compute_metrics`.

3. اتصل بـ [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model"،
...     remove_unused_columns=False,
...     eval_strategy="epoch"،
...     save_strategy="epoch"،
...     learning_rate=5e-5،
...     per_device_train_batch_size=16،
...     gradient_accumulation_steps=4،
...     per_device_eval_batch_size=16،
...     num_train_epochs=3،
...     warmup_ratio=0.1،
...     logging_steps=10،
...     load_best_model_at_end=True،
...     metric_for_best_model="accuracy"،
...     push_to_hub=True،
... )

>>> trainer = Trainer(
...     model=model،
...     args=training_args،
...     data_collator=data_collator،
...     train_dataset=food["train"]،
...     eval_dataset=food["test"]،
...     tokenizer=image_processor،
...     compute_metrics=compute_metrics،
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك مع المحور باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع [الدليل التعليمي الأساسي] (./training#train-a-tensorflow-model-with-keras) أولاً!

لضبط نموذج في TensorFlow، اتبع الخطوات التالية:

1. حدد فرط معلمات التدريب، وقم بإعداد محسن وجدول معدل التعلم.

2. قم بتنفيذ نموذج مُدرب مسبقًا.

3. قم بتحويل مجموعة بيانات 🤗 إلى `tf.data.Dataset`.

4. قم بتجميع نموذجك.

5. أضف استدعاءات الرجوع واستخدم طريقة `fit()` لتشغيل التدريب.

6. قم بتحميل نموذجك إلى 🤗 Hub لمشاركته مع المجتمع.

ابدأ بتحديد فرط المعلمات والمحسن وجدول معدل التعلم:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 5
>>> num_train_steps = len(food["train"]) * num_epochs
>>> learning_rate = 3e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate،
...     num_train_steps=num_train_steps،
...     weight_decay_rate=weight_decay_rate،
...     num_warmup_steps=0،
... )
```

ثم قم بتحميل ViT مع [`TFAutoModelForImageClassification`] جنبًا إلى جنب مع تخطيط التصنيفات:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained(
...     checkpoint،
...     id2label=id2label،
...     label2id=label2id،
... )
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~datasets.Dataset.to_tf_dataset`] و`data_collator` الخاص بك:

```py
>>> # تحويل مجموعة بيانات التدريب الخاصة بنا إلى tf.data.Dataset
>>> tf_train_dataset = food["train"].to_tf_dataset(
...     columns="pixel_values"، label_cols="label"، shuffle=True، batch_size=batch_size، collate_fn=data_collator
... )

>>> # تحويل مجموعة بيانات الاختبار الخاصة بنا إلى tf.data.Dataset
>>> tf_eval_dataset = food["test"].to_tf_dataset(
...     columns="pixel_values"، label_cols="label"، shuffle=True، batch_size=batch_size، collate_fn=data_collator
... )
```

قم بتهيئة النموذج للتدريب باستخدام `compile()`

```py
>>> from tensorflow.keras.losses import SparseCategoricalCrossentropy

>>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
>>> model.compile(optimizer=optimizer, loss=loss)
```

لحساب الدقة من التوقعات وتحميل نموذجك إلى 🤗 Hub، استخدم [Keras callbacks] (../main_classes/keras_callbacks).

مرر وظيفة `compute_metrics` الخاصة بك إلى [KerasMetricCallback] (../main_classes/keras_callbacks#transformers.KerasMetricCallback)،

واستخدم [PushToHubCallback] (../main_classes/keras_callbacks#transformers.PushToHubCallback) لتحميل النموذج:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="food_classifier"،
...     tokenizer=image_processor،
...     save_strategy="no"،
... )
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لتدريب نموذجك! اتصل بـ `fit()` باستخدام مجموعات بيانات التدريب والتحقق من الصحة الخاصة بك، وعدد الحقبات،

واستدعاءات الرجوع الخاصة بك لضبط النموذج:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
حقبة 1/5
250/250 [==============================] - 313s 1s/step - loss: 2.5623 - val_loss: 1.4161 - accuracy: 0.9290
حقبة 2/5
250/250 [==============================] - 265s 1s/step - loss: 0.9181 - val_loss: 0.6808 - accuracy: 0.9690
حقبة 3/5
250/250 [==============================] - 252s 1s/step - loss: 0.3910 - val_loss: 0.4303 - accuracy: 0.9820
حقبة 4/5
250/250 [==============================] - 251s 1s/step - loss: 0.2028 - val_loss: 0.3191 - accuracy: 0.9900
حقبة 5/5
250/250 [==============================] - 238s 949ms/step - loss: 0.1232 - val_loss: 0.3259 - accuracy: 0.9890
```

تهانينا! لقد ضبطت نموذجك وشاركته على 🤗 Hub. يمكنك الآن استخدامه للاستنتاج!

للحصول على مثال أكثر تفصيلاً حول كيفية ضبط نموذج لتصنيف الصور، راجع الدفتر المذكرة PyTorch المقابل [هنا] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

قم بتحميل صورة تريد تشغيل الاستدلال عليها:

```py
>>> ds = load_dataset("food101"، split="validation[:10]")
>>> image = ds["image"][0]
```

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه في [`pipeline`]. قم بتنفيذ `pipeline` لتصنيف الصور باستخدام نموذجك، ومرر صورتك إليه:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification"، model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224، 'label': 'beignets'}،
{'score': 0.015232225880026817، 'label': 'bruschetta'}،
{'score': 0.01519392803311348، 'label': 'chicken_wings'}،
{'score': 0.013022331520915031، 'label': 'pork_chop'}،
{'score': 0.012728818692266941، 'label': 'prime_rib'}]
```

يمكنك أيضًا يدويًا تكرار نتائج `pipeline` إذا أردت:

قم بتحميل معالج صورة لمعالجة الصورة وإرجاع `input` كرموز PyTorch:

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

مرر المدخلات إلى النموذج وأعد اللوغاريتمات:

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على التصنيف المتوقع مع أعلى احتمال، واستخدم تخطيط `id2label` للنموذج لتحويله إلى تصنيف:

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```

قم بتحميل معالج صورة لمعالجة الصورة وإرجاع `input` كرموز TensorFlow:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
>>> inputs = image_processor(image, return_tensors="tf")
```

مرر المدخلات إلى النموذج وأعد اللوغاريتمات:

```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
>>> logits = model(**inputs).logits
```

احصل على تصنيف متوقع بأعلى احتمال، واستخدم تخطيط `id2label` للنموذج لتحويله إلى تصنيف:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'beignets'
```