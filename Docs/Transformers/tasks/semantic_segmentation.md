## التقطيع الصوري

تفصل نماذج التقطيع الصوري المناطق التي تتوافق مع مناطق مختلفة ذات أهمية في صورة. تعمل هذه النماذج عن طريق تعيين تسمية لكل بكسل. هناك عدة أنواع من التقطيع: التقطيع الدلالي، وتقطيع المثيل، والتقطيع الشامل.

في هذا الدليل، سوف:

1. [إلقاء نظرة على أنواع مختلفة من التقطيع](#أنواع-التقطيع).
2. [لديك مثال شامل لضبط دقيق لتقطيع دلالي](#ضبط-دقيق-لنموذج-من-أجل-التقطيع).

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية
!pip install -q datasets transformers evaluate accelerate
```

نحن نشجعك على تسجيل الدخول إلى حسابك في Hugging Face حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## أنواع التقطيع

يعين التقطيع الدلالي تسمية أو فئة لكل بكسل في صورة. دعونا نلقي نظرة على إخراج نموذج التقطيع الدلالي. سوف يقوم بتعيين نفس الفئة لكل مثيل من كائن يصادفه في صورة، على سبيل المثال، سيتم تصنيف جميع القطط على أنها "قطة" بدلاً من "قطة-1"، "قطة-2".

يمكننا استخدام خط أنابيب التقطيع الصوري في المحولات للتنبؤ بسرعة بنموذج التقطيع الدلالي. دعونا نلقي نظرة على صورة المثال.

```python
from transformers import pipeline
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg" alt="إدخال التقطيع"/>
</div>

سنستخدم [nvidia/segformer-b1-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024).

```python
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)
results
```

يشمل إخراج خط أنابيب التقطيع قناعًا لكل فئة متوقعة.

```bash
[{'score': None,
'label': 'road',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'sidewalk',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'building',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'wall',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'pole',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'traffic sign',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'vegetation',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'terrain',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'sky',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': None,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

عند النظر إلى القناع لفئة السيارة، يمكننا أن نرى أن كل سيارة مصنفة بنفس القناع.

```python
results[-1]["mask"]
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/semantic_segmentation_output.png" alt="إخراج التقطيع الدلالي"/>
</div>

في تقطيع مثيل، الهدف ليس تصنيف كل بكسل، ولكن التنبؤ بقناع لكل **مثيل من كائن** في صورة معينة. إنه يعمل بشكل مشابه جدًا للكشف عن الأشياء، حيث يوجد مربع حد لكل مثيل، وهناك قناع تقطيع بدلاً من ذلك. سنستخدم [facebook/mask2former-swin-large-cityscapes-instance](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-instance) لهذا الغرض.

```python
instance_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-instance")
results = instance_segmentation(image)
results
```

كما ترون أدناه، هناك العديد من السيارات المصنفة، ولا يوجد تصنيف للبكسلات بخلاف البكسلات التي تنتمي إلى سيارة ومثيلات الأشخاص.

```bash
[{'score': 0.999944,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999945,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999652,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.903529,
'label': 'person',
'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

تفقد إحدى أقنعة السيارات أدناه.

```python
results[2]["mask"]
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/instance_segmentation_output.png" alt="إخراج التقطيع الدلالي"/>
</div>

يُدمج التقطيع الشامل بين التقطيع الدلالي وتقطيع المثيل، حيث يتم تصنيف كل بكسل إلى فئة ومثيل من تلك الفئة، وهناك أقنعة متعددة لكل مثيل من فئة. يمكننا استخدام [facebook/mask2former-swin-large-cityscapes-panoptic](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) لهذا الغرض.

```python
panoptic_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-panoptic")
results = panoptic_segmentation(image)
results
```

كما ترون أدناه، لدينا المزيد من الفئات. سنقوم لاحقًا بتوضيح أن كل بكسل مصنف إلى واحدة من الفئات.

```bash
[{'score': 0.999981,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999958,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.99997,
'label': 'vegetation',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999575,
'label': 'pole',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999958,
'label': 'building',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999634,
'label': 'road',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.996092,
'label': 'sidewalk',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.999221,
'label': 'car',
'mask': <PIL.Image.Image image mode=L size=612x415>},
{'score': 0.99987,
'label': 'sky',
'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

دعونا نقارن جنبا إلى جنب جميع أنواع التقطيع.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation-comparison.png" alt="خرائط التقطيع مقارنة"/>
</div>

بعد رؤية جميع أنواع التقطيع، دعونا نتعمق في ضبط دقيق لنموذج من أجل التقطيع الدلالي.

تشمل التطبيقات الواقعية الشائعة للتقطيع الدلالي تدريب السيارات ذاتية القيادة على التعرف على المشاة ومعلومات المرور المهمة، وتحديد الخلايا والتشوهات في الصور الطبية، ورصد التغيرات البيئية من صور الأقمار الصناعية.

## ضبط دقيق لنموذج من أجل التقطيع

سنقوم الآن بما يلي:

1. ضبط دقيق [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) على مجموعة بيانات [SceneParse150](https://huggingface.co/datasets/scene_parse_150).
2. استخدام نموذجك المضبوط بدقة للتنبؤ.

<Tip>

لرؤية جميع الهندسات ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/image-segmentation)

</Tip>
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين، مع تجاهل النصوص البرمجية وروابط الرموز.

### تحميل مجموعة بيانات SceneParse150

ابدأ بتحميل مجموعة فرعية أصغر من مجموعة بيانات SceneParse150 من مكتبة Datasets الخاصة بـ 🤗. سيتيح لك ذلك الفرصة للتجربة والتأكد من أن كل شيء يعمل قبل إنفاق المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

قم بتقسيم مجموعة البيانات المنقسمة على 'train' في مجموعة بيانات التدريب والاختبار باستخدام طريقة Dataset.train_test_split:

بعد ذلك، الق نظرة على مثال:

- "image": صورة PIL للمشهد.
- "annotation": صورة PIL لخريطة التجزئة، والتي تعد أيضًا هدف النموذج.
- "scene_category": معرف فئة يصف مشهد الصورة مثل "المطبخ" أو "المكتب". في هذا الدليل، ستحتاج فقط إلى "الصورة" و"التسمية التوضيحية"، وكلاهما صور PIL.

كما تريد إنشاء قاموس يقوم بتعيين معرف التسمية التوضيحية إلى فئة التسمية التوضيحية، والتي ستكون مفيدة عند إعداد النموذج لاحقًا. قم بتنزيل التعيينات من Hub وإنشاء القواميس id2label وlabel2id:

### مجموعة بيانات مخصصة

يمكنك أيضًا إنشاء مجموعة بياناتك الخاصة إذا كنت تفضل التدريب باستخدام البرنامج النصي run_semantic_segmentation.py بدلاً من مثيل دفتر الملاحظات. يتطلب البرنامج النصي ما يلي:

1. DatasetDict بـ Dataset مع عمودين Image، "image" و"label".

2. قاموس id2label يقوم بتعيين أعداد صحيحة للفئة إلى أسماء فئاتها.

وكمثال على ذلك، الق نظرة على مجموعة البيانات هذه التي تم إنشاؤها باستخدام الخطوات الموضحة أعلاه.

### معالجة مسبقة

الخطوة التالية هي تحميل معالج صور SegFormer لإعداد الصور والتعليقات التوضيحية للنموذج. تستخدم بعض مجموعات البيانات، مثل هذه، الفهرس صفر كفئة خلفية. ومع ذلك، فإن فئة الخلفية غير مدرجة بالفعل في 150 فئة، لذلك ستحتاج إلى تعيين do_reduce_labels=True لطرح واحد من جميع التسميات التوضيحية. يتم استبدال الفهرس صفر بـ 255 حتى يتم تجاهله بواسطة دالة الخسارة في SegFormer:

<frameworkcontent>
<pt>
من الشائع تطبيق بعض عمليات زيادة البيانات على مجموعة بيانات الصور لجعل النموذج أكثر قوة ضد الإفراط في التلائم. في هذا الدليل، ستستخدم دالة ColorJitter من torchvision لتغيير خصائص الألوان للصورة بشكل عشوائي، ولكن يمكنك أيضًا استخدام أي مكتبة صور تفضلها.

الآن، قم بإنشاء دالتين للمعالجة المسبقة لإعداد الصور والتعليقات التوضيحية للنموذج. تقوم هذه الدوال بتحويل الصور إلى "pixel_values" والتعليقات التوضيحية إلى "labels". بالنسبة لمجموعة بيانات التدريب، يتم تطبيق "jitter" قبل توفير الصور لمعالج الصور. بالنسبة لمجموعة الاختبار، يقوم معالج الصور باقتصاص وتطبيع "الصور"، ويقوم فقط باقتصاص "التسميات" لأن زيادة البيانات لا يتم تطبيقها أثناء الاختبار.

لتطبيق "jitter" على مجموعة البيانات بأكملها، استخدم وظيفة Dataset.set_transform من مكتبة Datasets:

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
من الشائع تطبيق بعض عمليات زيادة البيانات على مجموعة بيانات الصور لجعل النموذج أكثر قوة ضد الإفراط في التلائم. في هذا الدليل، ستستخدم وحدة tf.image لتغيير خصائص الألوان للصورة بشكل عشوائي، ولكن يمكنك أيضًا استخدام أي مكتبة صور تفضلها.

قم بتعريف دالتين للتحويل منفصلتين:

- تحويلات بيانات التدريب التي تتضمن زيادة الصور
- تحويلات بيانات التحقق التي تقوم فقط بترانزستور الصور، نظرًا لأن نماذج الرؤية الحاسوبية في 🤗 Transformers تتوقع تخطيط القنوات أولاً

قم بعد ذلك بإنشاء دالتين للمعالجة المسبقة لإعداد دفعات الصور والتعليقات التوضيحية للنموذج. تقوم هذه الدوال بتطبيق تحويلات الصور واستخدام معالج الصور المحمل سابقًا لتحويل الصور إلى "pixel_values" والتعليقات التوضيحية إلى "labels". كما يتولى معالج الصور أيضًا مسؤولية تغيير حجم الصور وتطبيعها.

لتطبيق تحويلات المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة Dataset.set_transform من مكتبة Datasets:

</tf>
</frameworkcontent>
### تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء النموذج. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة [Evaluate](https://huggingface.co/docs/evaluate/index) من 🤗 . بالنسبة لهذه المهمة، قم بتحميل مقياس [متوسط Intersection over Union](https://huggingface.co/spaces/evaluate-metric/accuracy) (IoU) (راجع الجولة السريعة من 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

ثم قم بإنشاء دالة لـ [`~evaluate.EvaluationModule.compute`] metrics. يجب تحويل تنبؤاتك إلى logits أولاً، ثم إعادة تشكيلها لمطابقة حجم التسميات قبل أن تتمكن من استدعاء [`~evaluate.EvaluationModule.compute`]:

<frameworkcontent>
<pt>

```py
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

```py
>>> def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = tf.transpose(logits, perm=[0, 2, 3, 1])
    logits_resized = tf.image.resize(
        logits,
        size=tf.shape(labels)[1:],
        method="bilinear",
    )

    pred_labels = tf.argmax(logits_resized, axis=-1)
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=num_labels,
        ignore_index=-1,
        reduce_labels=image_processor.do_reduce_labels,
    )

    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    return {"val_" + k: v for k, v in metrics.items()}
```

</tf>
</frameworkcontent>

الآن، أصبحت دالة `compute_metrics` الخاصة بك جاهزة للاستخدام، وستعود إليها عند إعداد التدريب.

### تدريب

<frameworkcontent>
<pt>

<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`،]، فراجع البرنامج التعليمي الأساسي [here](../training#finetune-with-trainer)!

</Tip>

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل SegFormer باستخدام [`AutoModelForSemanticSegmentation`]، ومرر إلى النموذج الخريطة بين معرفات التسميات وفئات التسميات:

```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. من المهم ألا تقوم بإزالة الأعمدة غير المستخدمة لأن هذا سيؤدي إلى إسقاط عمود "الصورة". بدون عمود "الصورة"، لا يمكنك إنشاء `pixel_values`. قم بتعيين `remove_unused_columns=False` لمنع هذا السلوك! الحجة المطلوبة الوحيدة الأخرى هي `output_dir` التي تحدد أين يتم حفظ نموذجك. ستقوم بالدفع بهذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم مقياس IoU وحفظ نقطة التحقق التدريبية.

2. مرر الحجج التدريبية إلى [`Trainer`] إلى جانب النموذج ومجموعة البيانات والمحلل اللغوي ومجمع البيانات و `compute_metrics` function.

3. استدعاء [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = TrainingArguments(
    output_dir="segformer-b0-scene-parse-150",
    learning_rate=6e-5,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=True,
)

>>> trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [basic tutorial](./training#train-a-tensorflow-model-with-keras) أولاً!

</Tip>

لضبط نموذج في TensorFlow، اتبع الخطوات التالية:

1. حدد فرط معلمات التدريب، وقم بإعداد محسن وجدول معدل التعلم.

2. قم بتحميل نموذج مسبق التدريب.

3. قم بتحويل مجموعة بيانات 🤗 إلى تنسيق `tf.data.Dataset`.

4. قم بتجميع نموذجك.

5. أضف استدعاءات للرجوع إلى الخلف لحساب المقاييس وتحميل نموذجك إلى 🤗 Hub

6. استخدم طريقة `fit()` لتشغيل التدريب.

ابدأ بتحديد فرط المعلمات والمحسن وجدول معدل التعلم:

```py
>>> from transformers import create_optimizer

>>> batch_size = 2
>>> num_epochs = 50
>>> num_train_steps = len(train_ds) * num_epochs
>>> learning_rate = 6e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0,
)
```

بعد ذلك، قم بتحميل SegFormer باستخدام [`TFAutoModelForSemanticSegmentation`] إلى جانب تعيينات التسميات، وقم بتجميعها باستخدام المحسن. لاحظ أن جميع نماذج Transformers تحتوي على دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)
>>> model.compile(optimizer=optimizer) # لا توجد حجة الخسارة!
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~datasets.Dataset.to_tf_dataset`] و [`DefaultDataCollator`]:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")

>>> tf_train_dataset = train_ds.to_tf_dataset(
    columns=["pixel_values", "label"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

>>> tf_eval_dataset = test_ds.to_tf_dataset(
    columns=["pixel_values", "label"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
```

لحساب الدقة من التنبؤات وتحميل نموذجك إلى 🤗 Hub، استخدم [Keras callbacks](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`KerasMetricCallback`]،
واستخدم [`PushToHubCallback`] لتحميل النموذج:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(
    metric_fn=compute_metrics, eval_dataset=tf_eval_dataset, batch_size=batch_size, label_cols=["labels"]
)

>>> push_to_hub_callback = PushToHubCallback(output_dir="scene_segmentation", tokenizer=image_processor)

>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لتدريب نموذجك! اتصل بـ `fit()` باستخدام مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور،
واستدعاءات الرجوع الخاصة بك لضبط النموذج:

```py
>>> model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=callbacks,
    epochs=num_epochs,
)
```

تهانينا! لقد ضبطت نموذجك وشاركته على 🤗 Hub. يمكنك الآن استخدامه للاستنتاج!

</tf>
</frameworkcontent>
### الاستنتاج
رائع، الآن بعد أن قمت بضبط نموذجك، يمكنك استخدامه للاستنتاج!

قم بإعادة تحميل مجموعة البيانات وتحميل صورة للاستنتاج.

```python
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
>>> ds = ds.train_test_split(test_size=0.2)
>>> test_ds = ds["test"]
>>> image = ds["test"][0]["image"]
>>> image
```

سنرى الآن كيفية الاستنتاج بدون خط أنابيب. قم بمعالجة الصورة باستخدام معالج الصور ووضع `pixel_values` على وحدة معالجة الرسوميات (GPU):

```python
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # استخدم وحدة معالجة الرسوميات إذا كانت متوفرة، وإلا استخدم وحدة المعالجة المركزية
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

مرر المدخلات إلى النموذج وأعد `logits`:

```python
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

بعد ذلك، قم بإعادة تحجيم `logits` إلى حجم الصورة الأصلي:

```python
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```

قم بتحميل معالج الصور لتحضير الصورة وإرجاع المدخلات على أنها تناظرات TensorFlow:

```python
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
>>> inputs = image_processor(image, return_tensors="tf")
```

مرر المدخلات إلى النموذج وأعد `logits`:

```python
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
>>> logits = model(**inputs).logits
```

بعد ذلك، قم بإعادة تحجيم `logits` إلى حجم الصورة الأصلي وقم بتطبيق `argmax` على البعد الطبقي:

```python
>>> logits = tf.transpose(logits, [0, 2, 3, 1])

>>> upsampled_logits = tf.image.resize(
...     logits,
...     # نعكس شكل `image` لأن `image.size` يعيد العرض والارتفاع.
...     image.size[::-1],
... )

>>> pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

لعرض النتائج، قم بتحميل لوحة ألوان مجموعة البيانات كما هو موضح في [ade_palette()] (https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) التي تقوم بماب كل فئة إلى قيم RGB الخاصة بها.

```python
def ade_palette():
    return np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 0],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])
```

بعد ذلك، يمكنك دمج خريطة الصورة وخريطة التجزئة المتوقعة وعرضهما:

```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1] # تحويل إلى BGR

>>> img = np.array(image) * 0.5 + color_seg * 0.5 # عرض الصورة مع خريطة التجزئة
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```