# الملخص 

الملخص هو إنشاء نسخة أقصر من المستند أو المقال الذي يلخص جميع المعلومات المهمة. إلى جانب الترجمة، يعد الملخص مثالًا آخر على المهام التي يمكن صياغتها كمهمة تسلسل إلى تسلسل. يمكن أن يكون الملخص:

- استخراجي: استخراج أهم المعلومات من المستند.
- تلخيصي: إنشاء نص جديد يلخص أهم المعلومات.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج T5 على مجموعة فرعية من فواتير ولاية كاليفورنيا في مجموعة بيانات BillSum للملخص التلخيصي.
2. استخدام نموذجك المضبوط للتنبؤ.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate rouge_score
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> من huggingface_hub استيراد notebook_login

>>> notebook_login()
```
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين فقط مع اتباع التعليمات التي قدمتها.

## تحميل مجموعة بيانات BillSum

ابدأ بتحميل الجزء الفرعي الأصغر من مجموعة بيانات BillSum لمقترحات ولاية كاليفورنيا من مكتبة Datasets:

## المعالجة المسبقة

الخطوة التالية هي تحميل محدد رموز T5 لمعالجة النص والملخص:

وظيفة المعالجة المسبقة التي تريد إنشائها تحتاج إلى:

1. إضافة بادئة إلى الإدخال مع عبارة "summarize: " حتى يعرف T5 أن هذه مهمة تلخيص. بعض النماذج القادرة على مهام NLP متعددة تتطلب إعطاء إشارة للمهام المحددة.
2. استخدام وسيط "text_target" عند تحديد رموز الملصقات.
3. اقطع التسلسلات بحيث لا تكون أطول من الطول الأقصى المحدد بواسطة معلمة "max_length".

لتطبيق وظيفة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة Dataset.map. يمكنك تسريع وظيفة "map" عن طريق تعيين "batched=True" لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

الآن، قم بإنشاء دفعة من الأمثلة باستخدام DataCollatorForSeq2Seq. من الأكثر كفاءة *التقسيم الديناميكي* للجمل إلى الطول الأطول في دفعة أثناء الجمع، بدلاً من تقسيم مجموعة البيانات بأكملها إلى الطول الأقصى.
## التقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء النموذج الخاص بك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة [Evaluate](https://huggingface.co/docs/evaluate/index) من 🤗 . بالنسبة لهذه المهمة، قم بتحميل مقياس [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```python
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتصنيفاتك إلى [`~evaluate.EvaluationModule.compute`] لحساب مقياس ROUGE:

```python
>>> import numpy as np

>>> def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
```

الآن، دالة `compute_metrics` الخاصة بك جاهزة للعمل، وستعود إليها عند إعداد التدريب الخاص بك.

## تدريب

<frameworkcontent>

<pt>

<Tip>

إذا لم تكن على دراية بتعديل نموذج باستخدام [`Trainer`]، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer) !

</Tip>

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل T5 باستخدام [`AutoModelForSeq2SeqLM`]:

```python
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`Seq2SeqTrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. سوف تقوم بدفع هذا النموذج إلى المحور عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم مقياس ROUGE وحفظ نقطة تفتيش التدريب.

2. قم بتمرير وسائط التدريب إلى [`Seq2SeqTrainer`] إلى جانب النموذج ومجموعة البيانات والمحلل اللغوي ومجمع البيانات ووظيفة `compute_metrics`.

3. استدعاء [`~Trainer.train`] لتعديل نموذجك.

```python
>>> training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

>>> trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على المحور باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```python
>>> trainer.push_to_hub()
```

</pt>

<tf>

<Tip>

إذا لم تكن على دراية بتعديل نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras) !

</Tip>

لتعديل نموذج في TensorFlow، ابدأ بإعداد دالة محسن ومعدل تعلم وجدول زمني وبعض فرط معلمات التدريب:

```python
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

بعد ذلك، يمكنك تحميل T5 باستخدام [`TFAutoModelForSeq2SeqLM`]:

```python
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```python
>>> tf_train_set = model.prepare_tf_dataset(
    tokenized_billsum["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

>>> tf_test_set = model.prepare_tf_dataset(
    tokenized_billsum["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
```

قم بتكوين النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers تحتوي على دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم تكن تريد ذلك:

```python
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer) # لا توجد وسيطة دالة الخسارة!
```

الأمران الأخيران اللذان يجب إعدادهما قبل بدء التدريب هما حساب درجة ROUGE من التنبؤات، وتوفير طريقة لدفع نموذجك إلى المحور. يتم تنفيذ كلاهما باستخدام [Keras callbacks](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```python
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد مكان دفع نموذجك ومحللك اللغوي في [`~transformers.PushToHubCallback`]:

```python
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
    output_dir="my_awesome_billsum_model",
    tokenizer=tokenizer,
)
```

بعد ذلك، قم بتجميع استدعاءاتك مرة أخرى:

```python
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت الآن على استعداد لبدء تدريب نموذجك! استدعاء [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحة البيانات، وعدد العصور، واستدعاءاتك لتعديل النموذج:

```python
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى المحور حتى يتمكن الجميع من استخدامه!

</tf>

</frameworkcontent>

<Tip>

للحصول على مثال أكثر تفصيلاً حول كيفية تعديل نموذج للتلخيص، راجع الدفتر المناسب
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb).

</Tip>
## الاستنتاج
رائع، الآن بعد أن ضبطت نموذجك، يمكنك استخدامه للاستنتاج!

قم بابتكار بعض النصوص التي ترغب في تلخيصها. بالنسبة لـ T5، يجب أن تقوم بإضافة بادئة إلى مدخلاتك بناءً على المهمة التي تعمل عليها. بالنسبة للتلخيص، يجب أن تقوم بإضافة بادئة إلى مدخلاتك كما هو موضح أدناه:

يكون استخدام نموذجك المضبوط للاستنتاج هو باستخدامه في ["pipeline"]. قم بتنفيذ عملية تلخيص باستخدام نموذجك، ومرر النص إليه:

يمكنك أيضًا محاكاة نتائج "pipeline" يدويًا إذا أردت:

قم برمجة النص واسترجاع "input_ids" كمصفوفات PyTorch:

استخدم طريقة ["~generation.GenerationMixin.generate"] لإنشاء التلخيص. لمزيد من التفاصيل حول استراتيجيات توليد النصوص المختلفة والمعلمات للتحكم في التوليد، راجع واجهة برمجة التطبيقات الخاصة بتوليد النصوص [Text Generation](../main_classes/text_generation).

قم بفك تشفير رموز العلامات المولدة مرة أخرى إلى نص:

قم برمجة النص واسترجاع "input_ids" كمصفوفات TensorFlow:

استخدم طريقة ["~transformers.generation_tf_utils.TFGenerationMixin.generate"] لإنشاء التلخيص. لمزيد من التفاصيل حول استراتيجيات توليد النصوص المختلفة والمعلمات للتحكم في التوليد، راجع واجهة برمجة التطبيقات الخاصة بتوليد النصوص [Text Generation](../main_classes/text_generation).

قم بفك تشفير رموز العلامات المولدة مرة أخرى إلى نص: