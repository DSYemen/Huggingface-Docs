# معالجة مسبقة

قبل أن تتمكن من تدريب نموذج على مجموعة بيانات، يجب معالجتها مسبقًا إلى تنسيق إدخال النموذج المتوقع. سواء كانت بياناتك نصية أو صورًا أو صوت، فيجب تحويلها إلى تنسورات وتجميعها في دفعات. توفر مكتبة 🤗 Transformers مجموعة من فئات المعالجة المسبقة للمساعدة في إعداد بياناتك للنموذج. في هذا البرنامج التعليمي، ستتعلم أنه:

* للنص، استخدم [راسم الأحرف](./main_classes/tokenizer) لتحويل النص إلى تسلسل من الرموز، وإنشاء تمثيل رقمي للرموز، وتجميعها في تنسورات.
* للكلام والصوت، استخدم [مستخرج الميزات](./main_classes/feature_extractor) لاستخراج ميزات متسلسلة من أشكال موجات الصوت وتحويلها إلى تنسورات.
* تستخدم إدخالات الصورة [معالج الصور](./main_classes/image_processor) لتحويل الصور إلى تنسورات.
* للإدخالات متعددة الوسائط، استخدم [معالج](./main_classes/processors) لدمج راسم أحرف ومستخرج ميزات أو معالج صور.

> تذكر أن `AutoProcessor` **يعمل دائمًا** ويختار تلقائيًا الفئة الصحيحة للنموذج الذي تستخدمه، سواء كنت تستخدم راسم أحرف أو معالج صور أو مستخرج ميزات أو معالج.

قبل البدء، قم بتثبيت 🤗 Datasets حتى تتمكن من تحميل بعض مجموعات البيانات لتجربتها:

```bash
pip install datasets
```

## معالجة اللغات الطبيعية

<Youtube id="Yffk5aydLzg"/>

أداة المعالجة المسبقة الرئيسية للبيانات النصية هي [راسم الأحرف](main_classes/tokenizer). يقسم راسم الأحرف النص إلى *رموز* وفقًا لمجموعة من القواعد. يتم تحويل الرموز إلى أرقام ثم إلى تنسورات، والتي تصبح إدخالات النموذج. ويضيف راسم الأحرف أي إدخالات إضافية مطلوبة من قبل النموذج.

> إذا كنت تخطط لاستخدام نموذج مُدرب مسبقًا، فمن المهم استخدام راسم الأحرف المُدرب مسبقًا المقترن به. يضمن ذلك تقسيم النص بنفس طريقة مجموعة بيانات التدريب، واستخدام نفس الرموز المقابلة للفهرس (يُشار إليها عادةً باسم *معجم*) أثناء التدريب المسبق.

ابدأ بتحميل راسم أحرف مُدرب مسبقًا باستخدام طريقة [`AutoTokenizer.from_pretrained`]. يقوم هذا الأسلوب بتنزيل *معجم* الذي تم تدريب النموذج عليه:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

ثم مرر النص إلى راسم الأحرف:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

يعيد راسم الأحرف قاموسًا يحتوي على ثلاثة عناصر مهمة:

* [input_ids](glossary#input-ids) هي الفهارس المقابلة لكل رمز في الجملة.
* [attention_mask](glossary#attention-mask) يشير إلى ما إذا كان يجب الاهتمام بالرمز أم لا.
* [token_type_ids](glossary#token-type-ids) يحدد التسلسل الذي ينتمي إليه الرمز عندما يكون هناك أكثر من تسلسل واحد.

أعد إدخالك عن طريق فك ترميز `input_ids`:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

كما ترى، أضاف راسم الأحرف رمزين خاصين - `CLS` و`SEP` (مصنف وفاصل) - إلى الجملة. لا تحتاج جميع النماذج إلى رموز خاصة، ولكن إذا كانت كذلك، فإن راسم الأحرف يضيفها تلقائيًا لك.

إذا كان هناك عدة جمل تريد معالجتها مسبقًا، فقم بتمريرها كقائمة إلى راسم الأحرف:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
[101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
[101, 1327, 1164, 5450, 23434, 136, 102]],
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0]],
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1]]}
```

### الحشو

ليست الجمل دائمًا بنفس الطول، والذي يمكن أن يكون مشكلة لأن التنسورات، إدخالات النموذج، يجب أن يكون لها شكل موحد. الحشو هو استراتيجية لضمان أن تكون التنسورات مستطيلة عن طريق إضافة رمز *حشو* خاص إلى الجمل الأقصر.

قم بتعيين معلمة `padding` إلى `True` لحشو الجمل الأقصر في الدفعة لمطابقة أطول تسلسل:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
[101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
[101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

تم حشو الجملتين الأولى والثالثة الآن بـ `0` لأنها أقصر.

### البتر

من ناحية أخرى، قد يكون التسلسل طويلًا جدًا بالنسبة للنموذج للتعامل معه. في هذه الحالة، ستحتاج إلى بتر التسلسل إلى طول أقصر.

قم بتعيين معلمة `truncation` إلى `True` لبتر تسلسل إلى الطول الأقصى الذي يقبله النموذج:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
[101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
[101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, Multiplier, 1, 1, 1, 1, 0، 0، 0، 0، 0، 0، 0، 0]]}
```

> تحقق من دليل المفاهيم [الحشو والبتر](./pad_truncation) لمعرفة المزيد حول حجة الحشو والبتر المختلفة.
## إنشاء المنسوجات

أخيرًا، تريد أن يقوم المحلل اللغوي بإرجاع المنسوجات الفعلية التي يتم تغذيتها إلى النموذج.

قم بتعيين معلمة "return_tensors" إما إلى "pt" لـ PyTorch، أو "tf" لـ TensorFlow:

<Tip>
تدعم الأنابيب المختلفة حجة المحلل اللغوي في "__call__()" بشكل مختلف. تدعم أنابيب "text-2-text-generation" (أي تمرير) فقط "truncation". تدعم أنابيب "text-generation" "max_length" و "truncation" و "padding" و "add_special_tokens".
في أنابيب "fill-mask"، يمكن تمرير حجة المحلل اللغوي في الحجة "tokenizer_kwargs" (قاموس).
</Tip>

## الصوت

بالنسبة للمهام الصوتية، ستحتاج إلى [مستخرج الميزات](main_classes/feature_extractor) لإعداد مجموعة البيانات الخاصة بك للنموذج. تم تصميم مستخرج الميزات لاستخراج الميزات من البيانات الصوتية الخام، وتحويلها إلى منسوجات.

قم بتحميل مجموعة بيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (راجع البرنامج التعليمي لـ 🤗 [Datasets](https://huggingface.co/docs/datasets/load_hub) للحصول على مزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيفية استخدام مستخرج الميزات مع مجموعات البيانات الصوتية:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

قم بالوصول إلى العنصر الأول من عمود "audio" لمعرفة المدخلات. يؤدي استدعاء عمود "audio" تلقائيًا إلى تحميل ملف الصوت وإعادة أخذ العينات منه:

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
0.        ,  0.        ], dtype=float32),
'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
'sampling_rate': 8000}
```

يعيد هذا ثلاثة عناصر:

* `array` هو إشارة الكلام المحملة - والتي تم إعادة أخذ عينات منها - كصفيف أحادي البعد.
* `path` يشير إلى موقع ملف الصوت.
* يشير `sampling_rate` إلى عدد نقاط البيانات في إشارة الكلام التي يتم قياسها في الثانية.

بالنسبة لهذا البرنامج التعليمي، ستستخدم نموذج [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base). الق نظرة على بطاقة النموذج، وستتعلم أن Wav2Vec2 مُدرب مسبقًا على الكلام الصوتي الذي تم أخذ عينات منه بمعدل 16 كيلو هرتز. من المهم أن تتطابق معدل العينات في بيانات الصوت مع معدل العينات في مجموعة البيانات المستخدمة لتدريب النموذج مسبقًا. إذا لم يكن معدل العينات الخاص ببياناتك هو نفسه، فيجب عليك إعادة أخذ عينات من بياناتك.

1. استخدم طريقة ["~datasets.Dataset.cast_column"] في 🤗 Datasets لإعادة أخذ العينات من معدل العينات إلى 16 كيلو هرتز:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. استدعاء عمود "audio" مرة أخرى لإعادة أخذ عينات من ملف الصوت:

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
'sampling_rate': 1600Multiplier: 16000,
'bits_per_sample': 32,
'format': 'wav',
'subtype': 'PCM_SIGNED'}
```

بعد ذلك، قم بتحميل مستخرج الميزات لتطبيع وإضافة وسائد المدخلات. عند إضافة وسائد إلى البيانات النصية، تتم إضافة "0" للتسلسلات الأقصر. تنطبق نفس الفكرة على البيانات الصوتية. يضيف مستخرج الميزات "0" - الذي يتم تفسيره على أنه صمت - إلى "array".

قم بتحميل مستخرج الميزات باستخدام ["AutoFeatureExtractor.from_pretrained"]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

مرر صفيف "audio" إلى مستخرج الميزات. كما نوصي بإضافة حجة "sampling_rate" في مستخرج الميزات من أجل تصحيح الأخطاء الصامتة التي قد تحدث بشكل أفضل.

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

ومثل المحلل اللغوي، يمكنك تطبيق الوسائد أو الاقتطاع للتعامل مع التسلسلات المتغيرة في دفعة. الق نظرة على طول التسلسل لهاتين العينتين الصوتيتين:

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

قم بإنشاء دالة لمعالجة مجموعة البيانات بحيث تكون عينات الصوت بنفس الأطوال. حدد طول عينة قصوى، وسيقوم مستخرج الميزات إما بإضافة وسائد أو اقتطاع التسلسلات لمطابقتها:

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

قم بتطبيق `preprocess_function` على أول بضع أمثلة في مجموعة البيانات:

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

أطوال العينات الآن متطابقة وتطابق الطول الأقصى المحدد. يمكنك الآن تمرير مجموعة البيانات المعالجة إلى النموذج!

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```
## رؤية الكمبيوتر

بالنسبة لمهام رؤية الكمبيوتر، ستحتاج إلى معالج صور لتحضير مجموعة البيانات الخاصة بك للنمذجة. تتكون المعالجة المسبقة للصور من عدة خطوات لتحويل الصور إلى المدخلات المتوقعة من النموذج. وتشمل هذه الخطوات، على سبيل المثال لا الحصر، تغيير الحجم، والتطبيع، وتصحيح قناة الألوان، وتحويل الصور إلى تنسورات.

غالبًا ما تتبع المعالجة المسبقة للصور بعض أشكال زيادة الصور. كل من المعالجة المسبقة للصور وزيادة الصور تحويل بيانات الصور، ولكنها تخدم أغراضًا مختلفة:

- تغيير الصورة: يعدل الصور بطريقة يمكن أن تساعد في منع الإفراط في التجهيز وزيادة متانة النموذج. يمكنك أن تكون مبدعًا في كيفية زيادة بياناتك - ضبط السطوع والألوان، والمحاصيل، والدوران، تغيير الحجم، التكبير، وما إلى ذلك. ومع ذلك، كن حذرًا من عدم تغيير معنى الصور باستخدام الزيادات الخاصة بك.

- المعالجة المسبقة للصور: تضمن أن الصور تتطابق مع تنسيق الإدخال المتوقع من النموذج. عند ضبط نموذج رؤية الكمبيوتر بدقة، يجب معالجة الصور بالضبط كما تم تدريب النموذج في البداية.

يمكنك استخدام أي مكتبة تريدها لزيادة الصور. بالنسبة للمعالجة المسبقة للصور، استخدم `ImageProcessor` المرتبط بالنموذج.

قم بتحميل مجموعة بيانات `food101` (راجع تعليمات `Datasets` للحصول على مزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيف يمكنك استخدام معالج الصور مع مجموعات بيانات رؤية الكمبيوتر:

استخدم معلمة `split` في `Datasets` لتحميل عينة صغيرة فقط من الانقسام التدريبي نظرًا لأن مجموعة البيانات كبيرة جدًا!

```بايثون
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

بعد ذلك، الق نظرة على الصورة باستخدام ميزة `Image` في `Datasets`:

```بايثون
>>> dataset[0]["image"]
```

قم بتحميل معالج الصور باستخدام `AutoImageProcessor.from_pretrained`:

```بايثون
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

أولاً، دعنا نضيف بعض الزيادة في الصور. يمكنك استخدام أي مكتبة تفضلها، ولكن في هذا البرنامج التعليمي، سنستخدم وحدة `transforms` من `torchvision`. إذا كنت مهتمًا باستخدام مكتبة زيادة بيانات أخرى، فتعرف على كيفية القيام بذلك في دفاتر ملاحظات `Albumentations` أو `Kornia`.

1. هنا نستخدم `Compose` لربط بعض التحويلات - `RandomResizedCrop` و`ColorJitter`. لاحظ أنه بالنسبة لتغيير الحجم، يمكننا الحصول على متطلبات حجم الصورة من `image_processor`. بالنسبة لبعض النماذج، من المتوقع ارتفاع وعرض محددين، وبالنسبة للآخرين، يتم تعريف `shortest_edge` فقط.

```بايثون
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. يقبل النموذج `pixel_values` كمدخلات له. يمكن لـ `ImageProcessor` التعامل مع تطبيع الصور، وتوليد التنسورات المناسبة. قم بإنشاء دالة تجمع بين زيادة الصور والمعالجة المسبقة للصور لمجموعة من الصور وتولد `pixel_values`:

```بايثون
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

في المثال أعلاه، قمنا بتعيين `do_resize=False` لأننا قمنا بالفعل بتغيير حجم الصور في تحويل زيادة الصور، واستفدنا من خاصية `size` من `image_processor` المناسب. إذا لم تقم بتغيير حجم الصور أثناء زيادة الصور، فقم بترك هذا المعلمة. بشكل افتراضي، سوف يتعامل `ImageProcessor` مع تغيير الحجم.

إذا كنت ترغب في تطبيع الصور كجزء من تحويل زيادة الصور، فاستخدم قيم `image_processor.image_mean` و`image_processor.image_std`.

3. ثم استخدم `datasets.Dataset.set_transform` لتطبيق التحويلات أثناء التنقل:

```بايثون
>>> dataset.set_transform(transforms)
```

4. الآن عند الوصول إلى الصورة، ستلاحظ أن معالج الصور قد أضاف `pixel_values`. يمكنك الآن تمرير مجموعة البيانات المعالجة إلى النموذج!

```بايثون
>>> dataset[0].keys()
```

هكذا تبدو الصورة بعد تطبيق التحويلات. تم اقتصاص الصورة عشوائيًا وتختلف خصائص الألوان بها.

```بايثون
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

بالنسبة للمهام مثل الكشف عن الأشياء، والتجزئة الدلالية، والتجزئة المثالية، والتجزئة الشاملة، يوفر `ImageProcessor` طرق ما بعد المعالجة. تحول هذه الطرق المخرجات الخام للنموذج إلى تنبؤات ذات معنى مثل صناديق الحدود، أو خرائط التجزئة.

## باد

في بعض الحالات، على سبيل المثال، عند ضبط نموذج `DETR` بدقة، يطبق النموذج زيادة المقياس في وقت التدريب. قد يتسبب هذا في اختلاف أحجام الصور في دفعة. يمكنك استخدام `DetrImageProcessor.pad` من `DetrImageProcessor` وتعريف `collate_fn` مخصص لدمج الصور معًا.

```بايثون
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## متعدد الوسائط

بالنسبة للمهام التي تتضمن مدخلات متعددة الوسائط، ستحتاج إلى معالج لتحضير مجموعة البيانات الخاصة بك للنمذجة. يقرن المعالج بين كائنين معالجين مثل المعالج والمحلل المميز.

قم بتحميل مجموعة بيانات `LJ Speech` (راجع تعليمات `Datasets` للحصول على مزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيف يمكنك استخدام معالج للتعرف التلقائي على الكلام (ASR):

```بايثون
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

بالنسبة لـ ASR، فأنت تركز بشكل أساسي على `audio` و`text` لذا يمكنك إزالة الأعمدة الأخرى:

```بايثون
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

الآن الق نظرة على أعمدة `audio` و`text`:

```بايثون
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

تذكر أنه يجب عليك دائمًا إعادة أخذ عينات من معدل عينات مجموعة بيانات الصوت الخاصة بك لمطابقة معدل العينات لمجموعة البيانات المستخدمة لتدريب النموذج مسبقًا!

```بايثون
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

قم بتحميل معالج باستخدام `AutoProcessor.from_pretrained`:

```بايثون
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. قم بإنشاء دالة لمعالجة بيانات الصوت الموجودة في `array` إلى `input_values`، ومعالجة `text` إلى `labels`. هذه هي المدخلات للنموذج:

```بايثون
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. قم بتطبيق دالة `prepare_dataset` على عينة:

```بايثون
>>> prepare_dataset(lj_speech[0])
```

لقد أضاف المعالج الآن `input_values` و`labels`، وتم أيضًا إعادة أخذ العينات من معدل العينات بشكل صحيح إلى 16 كيلو هرتز. يمكنك الآن تمرير مجموعة البيانات المعالجة إلى النموذج!