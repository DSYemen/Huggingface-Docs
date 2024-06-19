## الكشف عن الأشياء

الكشف عن الأشياء هو مهمة رؤية حاسوبية تهدف إلى اكتشاف مثيلات (مثل البشر أو المباني أو السيارات) في صورة. تتلقى نماذج الكشف عن الأشياء صورة كمدخلات وتخرج إحداثيات صناديق الحدود والعلامات المرتبطة بالأشياء المكتشفة. يمكن أن تحتوي الصورة على عدة أشياء، لكل منها صندوق حدوده وملصقه الخاص (على سبيل المثال، يمكن أن تحتوي على سيارة ومبنى)، ويمكن أن يوجد كل كائن في أجزاء مختلفة من الصورة (على سبيل المثال، يمكن أن تحتوي الصورة على عدة سيارات).

يُستخدم هذا التطبيق بشكل شائع في القيادة الذاتية للكشف عن أشياء مثل المشاة وإشارات الطرق وإشارات المرور. تشمل التطبيقات الأخرى حساب عدد الأشياء في الصور والبحث عن الصور والمزيد.

في هذا الدليل، ستتعلم كيفية:

1. ضبط نموذج DETR، وهو نموذج يجمع بين العمود الفقري التلافيفي ورمز المحول الترميزي، على مجموعة بيانات CPPE-5.
2. استخدام النموذج المضبوط دقيقًا للاستنتاج.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install -q datasets transformers accelerate timm
pip install -q -U albumentations>=1.4.5 torchmetrics pycocotools
```

ستستخدم Datasets لتحميل مجموعة بيانات من Hub Hugging Face، وTransformers لتدريب نموذجك، و`albumentations` لزيادة البيانات.

نحن نشجعك على مشاركة نموذجك مع المجتمع. قم بتسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميله إلى Hub.

عند المطالبة، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

للبدء، سنقوم بتعريف الثوابت العالمية، أي اسم النموذج وحجم الصورة. بالنسبة لهذا البرنامج التعليمي، سنستخدم نموذج DETR الشرطي بسبب تقاربه الأسرع. لا تتردد في اختيار أي نموذج للكشف عن الأشياء متاح في مكتبة "المحولات".

```py
>>> MODEL_NAME = "microsoft/conditional-detr-resnet-50" # أو "facebook/detr-resnet-50"
>>> IMAGE_SIZE = 480
```

## تحميل مجموعة بيانات CPPE-5

تحتوي مجموعة بيانات [CPPE-5](https://huggingface.co/datasets/cppe-5) على صور مع تعليقات توضيحية تحدد معدات الوقاية الشخصية الطبية (PPE) في سياق جائحة COVID-19.

ابدأ بتحميل مجموعة البيانات وإنشاء تقسيم "التحقق من الصحة" من "التدريب":

```py
>>> from datasets import load_dataset

>>> cppe5 = load_dataset("cppe-5")

>>> if "validation" not in cppe5:
...     split = cppe5["train"].train_test_split(0.15, seed=1337)
...     cppe5["train"] = split["train"]
...     cppe5["validation"] = split["test"]

>>> cppe5
DatasetDict({
train: Dataset({
features: ['image_id', 'image', 'width', 'height', 'objects'],
num_rows: 850
})
test: Dataset({
features: ['image_id', 'image', 'width', 'height', 'objects'],
num_rows: 29
})
validation: Dataset({
features: ['image_id', 'image', 'width', 'height', 'objects'],
num_rows: 150
})
})
```

سترى أن هذه المجموعة تحتوي على 1000 صورة لمجموعات التدريب والتحقق من الصحة ومجموعة اختبار بها 29 صورة.

للتعرف على البيانات، استكشف كيف تبدو الأمثلة.

```py
>>> cppe5["train"][0]
{
'image_id': 366,
'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=500x290>,
'width': 500,
'height': 500,
'objects': {
'id': [1932, 1933, 1934],
'area': [27063, 34200, 32431],
'bbox': [[29.0, 11.0, 97.0, 279.0],
[201.0, 1.0, 120.0, 285.0],
[382.0, 0.0, 113.0, 287.0]],
'category': [0, 0, 0]
}
}
```

تحتوي الأمثلة في مجموعة البيانات على الحقول التالية:

- `image_id`: معرف صورة المثال
- `image`: كائن `PIL.Image.Image` يحتوي على الصورة
- `width`: عرض الصورة
- `height`: ارتفاع الصورة
- `objects`: قاموس يحتوي على بيانات حدود مربع التعليق التوضيحي للأشياء الموجودة في الصورة:
  - `id`: معرف التعليق التوضيحي
  - `area`: مساحة صندوق الحدود
  - `bbox`: صندوق حدود الكائن (بتنسيق [COCO](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco))
  - `category`: فئة الكائن، مع القيم المحتملة بما في ذلك `Coverall (0)`، `Face_Shield (1)`، `Gloves (2)`، `Goggles (3)`، و`Mask (4)`

قد تلاحظ أن حقل "bbox" يتبع تنسيق COCO، وهو التنسيق الذي يتوقعه نموذج DETR. ومع ذلك، فإن تجميع الحقول داخل "الأشياء" يختلف عن تنسيق التعليق التوضيحي الذي يتطلبه DETR. ستحتاج إلى تطبيق بعض تحويلات المعالجة المسبقة قبل استخدام هذه البيانات للتدريب.

للحصول على فهم أفضل للبيانات، قم بتصور مثال في مجموعة البيانات.

```py
>>> import numpy as np
>>> import os
>>> from PIL import Image, ImageDraw

>>> image = cppe5["train"][2]["image"]
>>> annotations = cppe5["train"][2]["objects"]
>>> draw = ImageDraw.Draw(image)

>>> categories = cppe5["train"].features["objects"].feature["category"].names

>>> id2label = {index: x for index, x in enumerate(categories, start=0)}
>>> label2id = {v: k for k, v in id2label.items()}

>>> for i in range(len(annotations["id"])):
...     box = annotations["bbox"][i]
...     class_idx = annotations["category"][i]
...     x, y, w, h = tuple(box)
...     # تحقق مما إذا كانت الإحداثيات مقيدة أم لا
...     if max(box) > 1.0:
...         # الإحداثيات غير مقيدة، لا حاجة لإعادة تحجيمها
...         x1, y1 = int(x), int(y)
...         x2, y2 = int(x + w), int(y + h)
...     else:
...         # الإحداثيات مقيدة، إعادة تحجيمها
...         x1 = int(x * width)
...         y1 = int(y * height)
...         x2 = int((x + w) * width)
...         y2 = int((y + h) * height)
...     draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
...     draw.text((x, y), id2label[class_idx], fill="white")

>>> image
```

<div class="flex justify-center">
<img src="https://i.imgur.com/oVQb9SF.png" alt="مثال على مجموعة بيانات CPPE-5"/>
</div>

لعرض صناديق الحدود مع العلامات المرتبطة بها، يمكنك الحصول على العلامات من البيانات الوصفية لمجموعة البيانات، وتحديدًا حقل "الفئة".

كما تريد إنشاء القواميس التي تقوم بتعيين معرف العلامة إلى فئة العلامة (`id2label`) والعكس (`label2id`). يمكنك استخدامها لاحقًا عند إعداد النموذج. بما في ذلك هذه الخرائط سيجعل نموذجك قابلاً لإعادة الاستخدام من قبل الآخرين إذا قمت بمشاركته على Hugging Face Hub. يرجى ملاحظة أن جزء التعليمات البرمجية أعلاه الذي يرسم صناديق الحدود يفترض أنه في تنسيق `COCO` `(x_min، y_min، width، height)`. يجب تعديله للعمل بتنسيقات أخرى مثل `(x_min، y_min، x_max، y_max)`.

كخطوة نهائية للتعرف على البيانات، قم باستكشافها بحثًا عن مشكلات محتملة. تتمثل إحدى المشكلات الشائعة مع مجموعات البيانات للكشف عن الأشياء في صناديق الحدود التي "تمتد" إلى ما بعد حافة الصورة. يمكن أن تسبب صناديق الحدود "الهاربة" هذه أخطاء أثناء التدريب ويجب معالجتها. هناك بضع أمثلة تحتوي على هذه المشكلة في هذه المجموعة.

للحفاظ على البساطة في هذا الدليل، سنقوم بتعيين `clip=True` لـ `BboxParams` في التحولات أدناه.
بالتأكيد، سأتبع تعليماتك بدقة لترجمة النص التالي:

## معالجة البيانات مسبقًا

لضبط نموذج بدقة، يجب معالجة البيانات التي تخطط لاستخدامها لتتناسب بدقة مع النهج المستخدم للنموذج المُدرب مسبقًا.
يتولى [`AutoImageProcessor`] مهمة معالجة بيانات الصور لإنشاء `pixel_values` و`pixel_mask` و`labels` التي يمكن لنموذج DETR التدرب عليها. ولمعالج الصور بعض السمات التي لا داعي للقلق بشأنها:

- `image_mean = [0.485, 0.456, 0.406 ]`
- `image_std = [0.229, 0.224, 0.225]`

هذه هي المتوسطات والانحرافات المعيارية المستخدمة لتطبيع الصور أثناء التدريب المسبق للنموذج. هذه القيم بالغة الأهمية لتكرارها عند إجراء الاستدلال أو الضبط الدقيق لنموذج الصور المُدرب مسبقًا.

قم بتنفيذ معالج الصور من نفس نقطة التفتيش الخاصة بالنموذج الذي تريد ضبطه بدقة.

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained(
...     MODEL_NAME,
...     do_resize=True,
...     size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
...     do_pad=True,
...     pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
... )
```

قبل تمرير الصور إلى `image_processor`، قم بتطبيق تحويلين للمعالجة المسبقة على مجموعة البيانات:

- زيادة الصور
- إعادة تنسيق التعليقات لتلبية توقعات DETR

أولاً، للتأكد من أن النموذج لا يبالغ في التطابق مع بيانات التدريب، يمكنك تطبيق زيادة الصور باستخدام أي مكتبة لزيادة البيانات. هنا نستخدم [Albumentations](https://albumentations.ai/docs/).
تضمن هذه المكتبة أن تؤثر التحويلات على الصورة وتحدِّث صناديق الحدود وفقًا لذلك.
تحتوي وثائق مكتبة مجموعات البيانات على دليل تفصيلي حول [كيفية زيادة الصور للكشف عن الأشياء](https://huggingface.co/docs/datasets/object_detection)،
ويستخدم نفس مجموعة البيانات كمثال. قم بتطبيق بعض التحولات الهندسية واللونية على الصورة. لمزيد من خيارات الزيادة، استكشف [مساحة عرض Albumentations](https://huggingface.co/spaces/qubvel-hf/albumentations-demo).

```py
>>> import albumentations as A

>>> max_size = IMAGE_SIZE

>>> train_augment_and_transform = A.Compose(
...     [
...         A.Perspective(p=0.1),
...         A.HorizontalFlip(p=0.5),
...         A.RandomBrightnessContrast(p=0.5),
...         A.HueSaturationValue(p=0.1),
...     ],
...     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
... )

>>> validation_transform = A.Compose(
...     [A.NoOp()],
...     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
... )
```

يتوقع `image_processor` أن تكون التعليقات بالشكل التالي: `{'image_id': int, 'annotations': List[Dict]}`،
حيث كل قاموس هو تعليق كائن COCO. دعنا نضيف دالة لإعادة تنسيق التعليقات لمثال واحد:

```py
>>> def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
...     """قم بتنسيق مجموعة واحدة من تعليقات الصور بتنسيق COCO

...     Args:
...         image_id (str): معرف الصورة. على سبيل المثال، "0001"
...         categories (List[int]): قائمة بالفئات/علامات الفئات المقابلة لصناديق الحدود المقدمة
...         areas (List[float]): قائمة بالمساحات المقابلة لصناديق الحدود المقدمة
...         bboxes (List[Tuple[float]]): قائمة بصناديق الحدود المقدمة بتنسيق COCO
...             ([center_x, center_y, width, height] في الإحداثيات المطلقة)

...     Returns:
...         dict: {
...             "image_id": معرف الصورة،
...             "annotations": قائمة بالتعليقات المنسقة
...         }
...     """
...     annotations = []
...     لكل فئة، مساحة، صندوق حدود في الرمز البريدي:
...         التعليق المنسق = {
...             "image_id": image_id،
...             "category_id": category,
...             "iscrowd": 0,
...             "area": area,
...             "bbox": list(bbox),
...         }
...         annotations.append(formatted_annotation)

...     return {
...         "image_id": image_id,
...         "annotations": annotations,
...     }

```

الآن يمكنك الجمع بين تحويلات الصور والتعليقات لاستخدامها في دفعة من الأمثلة:

```py
>>> def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
...     """تطبيق الزيادات وتنسيق التعليقات بتنسيق COCO لمهمة الكشف عن الأشياء"""

...     images = []
...     annotations = []
...     لكل معرف صورة، صورة، كائنات في الرمز البريدي:
...         image = np.array(image.convert("RGB"))

...         # تطبيق الزيادات
...         output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
...         images.append(output["image"])

...         # تنسيق التعليقات بتنسيق COCO
...         formatted_annotations = format_image_annotations_as_coco(
...             image_id, output["category"], objects["area"], output["bboxes"]
...         )
...         annotations.append(formatted_annotations)

...     # تطبيق تحويلات معالج الصور: تغيير الحجم، إعادة التقييم، التطبيع
...     result = image_processor(images=images, annotations=annotations, return_tensors="pt")

...     if not return_pixel_mask:
...         result.pop("pixel_mask", None)

...     return result
```

قم بتطبيق دالة المعالجة المسبقة هذه على مجموعة البيانات بأكملها باستخدام طريقة [`~datasets.Dataset.with_transform`] في مكتبة مجموعات البيانات. تطبق هذه الطريقة
التحويلات أثناء التنقل عند تحميل عنصر من مجموعة البيانات.

في هذه المرحلة، يمكنك التحقق من الشكل الذي يبدو عليه مثال من مجموعة البيانات بعد التحويلات. يجب أن ترى tensor
مع `pixel_values`، tensor مع `pixel_mask`، و`labels`.

```py
>>> from functools import partial

>>> # إنشاء دالات تحويل للدفعة وتطبيقها على أقسام مجموعة البيانات
>>> train_transform_batch = partial(
...     augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
... )
>>> validation_transform_batch = partial(
...     augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
... )

>>> cppe5["train"] = cppe5["train"].with_transform(train_transform_batch)
>>> cppe5["validation"] = cppe5["validation"].with_transform(validation_transform_batch)
>>> cppe5["test"] = cppe5["test"].with_transform(validation_transform_batch)

>>> cppe5["train"][15]
{'pixel_values': tensor([[[ 1.9235,  1.9407,  1.9749,  ..., -0.7822, -0.7479, -0.6965],
[ 1.9578,  1.9749,  1.9920,  ..., -0.7993, -0.7650, -0.7308],
[ 2.0092,  2.0092,  2.0263,  ..., -0.8507, -0.8164, -0.7822],
...,
[ 0.0741,  0.0741,  0.0741,  ...,  0.0741,  0.0741,  0.0741],
[ 0.0741,  0.0741,  0.0741,  ...,  0.0741,  0.0741,  0.0741],
[ 0.0741,  0.0741,  0.0741,  ...,  0.0741,  0.0741,  0.0741]],

[[ 1.6232,  1.6408,  1.6583,  ...,  0.8704,  1.0105,  1.1331],
[ 1.6408,  1.6583,  1.6758,  ...,  0.8529,  0.9930,  1.0980],
[ 1.6933,  1.6933,  1.7108,  ...,  0.8179,  0.9580,  1.0630],
...,
[ 0.2052,  0.2052,  0.2052,  ...,  0.2052,  0.2052,  0.2052],
[ 0.2052,  0.2052,  0.2052,  ...,  0.2052,  0.2052,  0.2052],
[ 0.2052,  0.2052,  0.2052,  ...,  0.2052,  0.2052,  0.2052]],

[[ 1.8905,  1.9080,  1.9428,  ..., -0.1487, -0.0964, -0.0615],
[ 1.9254,  1.9428,  1.9603,  ..., -0.1661, -0.1138, -0.0790],
[ 1.9777,  1.9777,  1Multiplier,  ..., -0.2010, -0.1138, -0.0790],
...,
[ 0.4265,  0.4265,  0.4265,  ...,  0.4265,  0.4265,  0.4265],
[ 0.4265,  0.4265,  0.4265,  ...,  0.4265,  0.4265,  0.4265],
[ 0.4265,  0.4265,  0.4265,  ...,  0.4265,  0.4265,  0.4265]]]),
'labels': {'image_id': tensor([688]), 'class_labels': tensor([3, 4, 2, 0, 0]), 'boxes': tensor([[0.4700, 0.1933, 0.1467, 0.0767],
[0.4858, 0.2600, 0.1150, 0.1000],
[0.4042, 0.4517, 0.1217, 0.1300],
[0.4242, 0.3217, 0.3617, 0.5567],
[0.6617, 0.4033, 0.5400, 0.4533]]), 'area': tensor([ 4048.,  4140.,  5694., 72478., 88128.]), 'iscrowd': tensor([0, 0, 0, 0, 0]), 'orig_size': tensor([480, 480])}}
```

لقد نجحت في زيادة الصور الفردية وإعداد تعليقاتها. ومع ذلك، فإن المعالجة المسبقة
لم تكتمل بعد. في الخطوة الأخيرة، قم بإنشاء `collate_fn` مخصص لدمج الصور في دفعات.
قم بتبديل الصور (التي هي الآن `pixel_values`) إلى أكبر صورة في دفعة، وقم بإنشاء `pixel_mask`
لتوضيح البكسلات الحقيقية (1) والبكسلات المضافة (0).

```py
>>> import torch

>>> def collate_fn(batch):
...     data = {}
...     data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
...     data["labels"] = [x["labels"] for x in batch]
...     if "pixel_mask" in batch[0]:
...         data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
...     return data

```
## إعداد الدالة لحساب mAP

تُقيَّم نماذج اكتشاف الأشياء عادةً باستخدام مجموعة من المقاييس على طريقة COCO. سنستخدم torchmetrics لحساب مقاييس mAP (متوسط الدقة المتوسطة) وmAR (متوسط الاستدعاء المتوسط) وسنقوم بتغليفها في دالة compute_metrics حتى نتمكن من استخدامها في Trainer للتقييم.

تُستخدم تنسيقات YOLO (المُعاد تحجيمها) كصيغة وسيطة للصناديق المستخدمة في التدريب، ولكننا سنحسب المقاييس للصناديق في تنسيق Pascal VOC (المطلق) من أجل التعامل مع مناطق الصندوق بشكل صحيح. دعونا نحدد دالة تحول صناديق الإحداثيات إلى تنسيق Pascal VOC:

ثم في دالة compute_metrics، نقوم بجمع الصناديق والدرجات والعلامات المتوقعة والمستهدفة من نتائج حلقة التقييم ونمررها إلى دالة التقييم.

دالة حساب المقاييس:

```

تأخذ هذه الدالة نتائج التقييم على شكل تنبؤات وأهداف، بالإضافة إلى عتبة اختيارية لتصفية الصناديق المتوقعة بناءً على مستوى الثقة. تقوم الدالة بمعالجة التنبؤات والأهداف لتتوافق مع الصيغة المطلوبة لحساب المقاييس، ثم تستخدم دالة MeanAveragePrecision لحساب المقاييس المختلفة. أخيرًا، يتم تنسيق النتائج في شكل قاموس يحتوي على أسماء المقاييس وقيمها.

في النهاية، نقوم بإنشاء دالة eval_compute_metrics_fn باستخدام الدالة الجزئية، والتي تقوم بتمرير وسيطات إضافية إلى دالة compute_metrics.
## تدريب نموذج الكشف

لقد قمت بمعظم العمل الشاق في الأقسام السابقة، لذا فأنت الآن مستعد لتدريب نموذجك!

لا تزال الصور في مجموعة البيانات هذه كبيرة جدًا، حتى بعد تغيير حجمها. وهذا يعني أن ضبط هذا النموذج سيتطلب وحدة معالجة رسومات واحدة على الأقل.

يتضمن التدريب الخطوات التالية:

1. قم بتحميل النموذج باستخدام [`AutoModelForObjectDetection`] باستخدام نفس نقطة التفتيش كما في المعالجة المسبقة.
2. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`].
3. قم بتمرير فرط معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات ومعالج الصور ومجمع البيانات.
4. استدعاء [`~Trainer.train`] لضبط نموذجك.

عند تحميل النموذج من نفس نقطة التفتيش التي استخدمتها للمعالجة المسبقة، تذكر تمرير الخرائط `label2id` و `id2label` التي أنشأتها سابقًاً من البيانات الوصفية لمجموعة البيانات. بالإضافة إلى ذلك، نحدد `ignore_mismatched_sizes=True` لاستبدال رأس التصنيف الموجود برأس جديد.

```py
>>> from transformers import AutoModelForObjectDetection

>>> model = AutoModelForObjectDetection.from_pretrained(
...     MODEL_NAME,
...     id2label=id2label,
...     label2id=label2id,
...     ignore_mismatched_sizes=True,
... )
```

في [`TrainingArguments`] استخدم `output_dir` لتحديد مكان حفظ نموذجك، ثم قم بتكوين فرط المعلمات كما تراه مناسبًا. بالنسبة إلى `num_train_epochs=30`، سيستغرق التدريب حوالي 35 دقيقة في Google Colab T4 GPU، قم بزيادة عدد الفترات لتحقيق نتائج أفضل.

ملاحظات مهمة:

- لا تقم بإزالة الأعمدة غير المستخدمة لأن هذا سيؤدي إلى إسقاط عمود الصورة. بدون عمود الصورة، لا يمكنك إنشاء `pixel_values`. لهذا السبب، قم بتعيين `remove_unused_columns` إلى `False`.
- قم بتعيين `eval_do_concat_batches=False` للحصول على نتائج تقييم صحيحة. تحتوي الصور على عدد مختلف من صناديق الهدف، إذا تم دمج الدفعات، فلن نتمكن من تحديد الصناديق التي تنتمي إلى صورة معينة.

إذا كنت ترغب في مشاركة نموذجك عن طريق دفعه إلى Hub، فقم بتعيين `push_to_hub` إلى `True` (يجب أن تكون قد سجلت الدخول إلى Hugging Face لتحميل نموذجك).

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(
...     output_dir="detr_finetuned_cppe5"،
...     num_train_epochs=30،
...     fp16=False،
...     per_device_train_batch_size=8،
...     dataloader_num_workers=4،
...     learning_rate=5e-5،
...     lr_scheduler_type="cosine"،
...     weight_decay=1e-4،
...     max_grad_norm=0.01،
...     metric_for_best_model="eval_map"،
...     greater_is_better=True،
...     load_best_model_at_end=True،
...     eval_strategy="epoch"،
...     save_strategy="epoch"،
...     save_total_limit=2،
...     remove_unused_columns=False،
...     eval_do_concat_batches=False،
...     push_to_hub=True،
... )
```

أخيرًا، قم بتجميع كل شيء معًا، واستدعاء [`~transformers.Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=cppe5["train"],
...     eval_dataset=cppe5["validation"],
...     tokenizer=image_processor,
...     data_collator=collate_fn,
...     compute_metrics=eval_compute_metrics_fn,
... )

>>> trainer.train()
```

<div>
<progress value='3210' max='3210' style='width:300px; height:20px; vertical-align: middle;'></progress>
[3210/3210 26:07، الفصل 30/30]
</div>

<table border="1" class="dataframe">
<thead>
<tr style="text-align: left;">
<th>الفصل</th>
<th>فقدان التدريب</th>
<th>خسارة التحقق</th>
<th>خريطة</th>
<th>خريطة 50</th>
<th>خريطة 75</th>
<th>خريطة صغيرة</th>
<th>خريطة متوسطة</th>
<th>خريطة كبيرة</th>
<th>مار 1</th>
<th>مار 10</th>
<th>مار 100</th>
<th>مار صغير</th>
<th>مار متوسط</th>
<th>مار كبير</th>
<th>خريطة كوفيرال</th>
<th>مار 100 كوفيرال</th>
<th>خريطة درع الوجه</th>
<th>مار 100 درع الوجه</th>
<th>خريطة القفازات</th>
<th>مار 100 قفازات</th>
<th>خريطة النظارات</th>
<th>مار 100 نظارات</th>
<th>خريطة القناع</th>
<th>مار 100 قناع</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>لا يوجد سجل</td>
<td>2.629903</td>
<td>0.008900</td>
<td>0.023200</td>
<td>0.006500</td>
<td>0.001300</td>
<td>0.002800</td>
<td>0.020500</td>
<td>0.021500</td>
<td>0.070400</td>
<td>0.101400</td>
<td>0.007600</td>
<td>0.106200</td>
<td>0.036700</td>
<td>0.232000</td>
<td>0.000300</td>
<td>0.019000</td>
<td>0.003900</td>
<td>0.125400</td>
<td>0.000100</td>
<td>0.003100</td>
<td>0.003500</td>
<td>0.127600</td>
</tr>
<tr>
<td>2</td>
<td>لا يوجد سجل</td>
<td>3.479864</td>
<td>0.014800</td>
<td>0.034600</td>
<td>0.010800</td>
<td>0.008600</td>
<td>0.011700</td>
<td>0.012500</td>
<td>0.041100</td>
<td>0.098700</td>
<td>0.130000</td>
<td>0.056000</td>
<td>0.062200</td>
<td>0.111900</td>
<td>0.053500</td>
<td>0.447300</td>
<td>0.010600</td>
<td>0.100000</td>
<td>0.000200</td>
<td>0.022800</td>
<td>0.000100</td>
<td>0.015400</td>
<td>0.009700</td>
<td>0.064400</td>
</tr>
<tr>
<td>3</td>
<td>لا يوجد سجل</td>
<td>2.107622</td>
<td>0.041700</td>
<td>0.094000</td>
<td>0.034300</td>
<td>0.024100</td>
<td>0.026400</td>
<td>0.047400</td>
<td>0.091500</td>
<td>0.182800</td>
<td>0.225800</td>
<td>0.087200</td>
<td>0.199400</td>
<td>0.210600</td>
<td>0.150900</td>
<td>0.571200</td>
<td>0.017300</td>
<td>0.101300</td>
<td>0.007300</td>
<td>0.180400</td>
<td>0.002100</td>
<td>0.026200</td>
<td>0.031000</td>
<td>0.250200</td>
</tr>
<tr>
<td>4</td>
<td>لا يوجد سجل</td>
<td>2.031242</td>
<td>0.055900</td>
<td>0.120600</td>
<td>0.046900</td>
<td>0.013800</td>
<td>0.038100</td>
<td>0.090300</td>
<td>0.105900</td>
<td>0.225600</td>
<td>0.266100</td>
<td>0.130200</td>
<td>0.228100</td>
<td>0.330000</td>
<td>0.191000</td>
<td>0.572100</td>
<td>0.010600</td>
<td>0.157000</td>
<td>0.014600</td>
<td>0.235300</td>
<td>0.001700</td>
<td>0.052300</td>
<td>0.061800</td>
<td>0.313800</td>
</tr>
<tr>
<td>5</td>
<td>3.889400</td>
<td>1.883433</td>
<td>0.089700</td>
<td>0.201800</td>
<td>0.067300</td>
<td>0.022800</td>
<td>0.065300</td>
<td>0.129500</td>
<td>0.136000</td>
<td>0.272200</td>
<td>0.303700</td>
<td>0.112900</td>
<td>0.312500</td>
<td>0.424600</td>
<td>0.300200</td>
<td>0.585100</td>
<td>0.032700</td>
<td>0.202500</td>
<td>0.031300</td>
<td>0.271000</td>
<td>0.008700</td>
<td>0.126200</td>
<td>0.075500</td>
<td>0.333800</td>
</tr>
<tr>
<td>6</td>
<td>3.889400</td>
<td>1.807503</td>
<td>0.118500</td>
<td>0.270900</td>
<td>0.090200</td>
<td>0.034900</td>
<td>0.076700</td>
<td>0.152500</td>
<td>0.146100</td>
<td>0.297800</td>
<td>0.325400</td>
<td>0.171700</td>
<td>0.283700</td>
<td>0.545900</td>
<td>0.396900</td>
<td>0.554500</td>
<td>0.043000</td>
<td>0.262000</td>
<td>0.054500</td>
<td>0.271900</td>
<td>0.020300</td>
<td>0.230800</td>
<td>0.077600</td>
<td>0.308000</td>
</tr>
<tr>
<td>7</td>
<td>3.889400</td>
<td>1.716169</td>
<td>0.143500</td>
<td>0.307700</td>
<td>0.123200</td>
<td>0.045800</td>
<td>0.097800</td>
<td>0.258300</td>
<td>0.165300</td>
<td>0.327700</td>
<td>0.352600</td>
<td>0.140900</td>
<td>0.336700</td>
<td>0.599400</td>
<td>0.442900</td>
<td>0.620700</td>
<td>0.069400</td>
<td>0.301300</td>
<td>0.081600</td>
<td>0.292000</td>
<td>0.011000</td>
<td>0.230800</td>
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين فقط مع الحفاظ على تنسيق Markdown:

## التقييم

تعطي هذه النتائج لمحة عن أداء النموذج على مجموعة الاختبار. يمكن تحسين هذه النتائج عن طريق ضبط فرط المعلمات في [`TrainingArguments`]. جرّب ذلك!

## الاستنتاج

الآن بعد أن قمت بضبط نموذج، وتقييمه، وتحميله على Hugging Face Hub، يمكنك استخدامه للاستنتاج.

قم بتحميل الصورة من عنوان URL:

## الاستنتاج

الآن بعد أن قمت بضبط نموذج، وتقييمه، وتحميله على Hugging Face Hub، يمكنك استخدامه للاستنتاج.

قم بتحميل الصورة من عنوان URL:

قم بتحميل النموذج ومعالج الصور من Hugging Face Hub (تخطي هذا الجزء إذا كنت قد قمت بتدريب النموذج بالفعل في هذه الجلسة):

والآن، قم باكتشاف حدود الصناديق:

دعونا نرسم النتيجة:

<div class="flex justify-center">
<img src="https://i.imgur.com/oDUqD0K.png" alt="نتيجة الكشف عن الكائنات على صورة جديدة"/>
</div>