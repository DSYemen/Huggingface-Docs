# تحميل مثيلات مُدربة مسبقًا باستخدام AutoClass

مع وجود العديد من بنى Transformer المختلفة، يمكن أن يكون إنشاء واحدة منها لنقطة تفتيش معينة أمرًا صعبًا. وكجزء من الفلسفة الأساسية لمكتبة 🤗 Transformers لجعلها سهلة وبسيطة ومرنة، تقوم فئة `AutoClass` تلقائيًا باستنتاج وتحميل البنية الصحيحة من نقطة تفتيش معينة. وتسمح طريقة `from_pretrained()` بتحميل نموذج مُدرب مسبقًا لأي بنية بسرعة بحيث لا يتعين عليك تكريس الوقت والموارد لتدريب نموذج من الصفر. ويعني إنتاج هذا النوع من الرموز المرجعية غير المعتمدة على نقاط التفتيش أنه إذا كان رمزك يعمل لنقطة تفتيش واحدة، فسيتم تشغيله مع نقطة تفتيش أخرى - طالما تم تدريبه لمهمة مماثلة - حتى إذا كانت البنية مختلفة.

<Tip>

تذكر أن البنية تشير إلى هيكل النموذج، بينما الرموز المرجعية هي الأوزان لبنية معينة. على سبيل المثال، BERT هي بنية، في حين أن `google-bert/bert-base-uncased` هي نقطة تفتيش. والنموذج هو مصطلح عام يمكن أن يعني البنية أو نقطة التفتيش.

</Tip>

في هذا البرنامج التعليمي، ستتعلم كيفية:

- تحميل رموز مُدربة مسبقًا
- تحميل معالج صور مُدرب مسبقًا
- تحميل مستخرج ميزات مُدرب مسبقًا
- تحميل معالج مُدرب مسبقًا
- تحميل نموذج مُدرب مسبقًا
- تحميل نموذج كعمود فقري

## AutoTokenizer

تبدأ معظم مهام معالجة اللغات الطبيعية بمُرمز يحول المدخلات إلى تنسيق يمكن للنموذج معالجته.

قم بتحميل مُرمز باستخدام [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

ثم قم برمز تسلسل المدخلات كما هو موضح أدناه:

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

بالنسبة لمهام الرؤية، يقوم معالج الصور بمعالجة الصورة إلى تنسيق الإدخال الصحيح.

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

## AutoBackbone

<div style="text-align: center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stages.png">
<figcaption class="mt-2 text-center text-sm text-gray-500">عمود فقري Swin مع مراحل متعددة لإخراج خريطة سمات.</figcaption>
</div>

تتيح لك [`AutoBackbone`] استخدام النماذج المُدربة مسبقًا كعمود فقري للحصول على خرائط سمات من مراحل مختلفة من العمود الفقري. يجب عليك تحديد أحد المعلمات التالية في [`~PretrainedConfig.from_pretrained`]:

- `out_indices` هو فهرس الطبقة التي تريد الحصول على خريطة السمات منها
- `out_features` هو اسم الطبقة التي تريد الحصول على خريطة السمات منها

يمكن استخدام هذه المعلمات بشكل متبادل، ولكن إذا استخدمت كلتيهما، فتأكد من مواءمتهما! إذا لم تمرر أيًا من هذه المعلمات، فسيقوم العمود الفقري بإرجاع خريطة السمات من الطبقة الأخيرة.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stage%201.png">
<figcaption class="mt-2 text-center text-sm text-gray-500">خريطة سمات من المرحلة الأولى للعمود الفقري. يشير تقسيم الرقع إلى جذع النموذج.</figcaption>
</div>

على سبيل المثال، في الرسم التخطيطي أعلاه، لإرجاع خريطة السمات من المرحلة الأولى للعمود الفقري Swin، يمكنك تعيين `out_indices=(1,)`:

```py
>>> from transformers import AutoImageProcessor, AutoBackbone
>>> import torch
>>> from PIL import Image
>>> import requests
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))

>>> inputs = processor(image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> feature_maps = outputs.feature_maps
```

الآن يمكنك الوصول إلى كائن `feature_maps` من المرحلة الأولى للعمود الفقري:

```py
>>> list(feature_maps[0].shape)
[1, 96, 56, 56]
```

## AutoFeatureExtractor

بالنسبة لمهام الصوت، يقوم مستخرج الميزات بمعالجة إشارة الصوت إلى تنسيق الإدخال الصحيح.

قم بتحميل مستخرج ميزات باستخدام [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

تتطلب المهام متعددة الوسائط معالجًا يجمع بين نوعين من أدوات المعالجة المسبقة. على سبيل المثال، يتطلب نموذج [LayoutLMV2](model_doc/layoutlmv2) معالج صور للتعامل مع الصور ومُرمز للتعامل مع النص؛ يجمع المعالج بين الاثنين.

قم بتحميل معالج باستخدام [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

<frameworkcontent>
<pt>
تتيح لك فئات `AutoModelFor` تحميل نموذج مُدرب مسبقًا لمهمة معينة (راجع [هنا](model_doc/auto) للحصول على قائمة كاملة بالمهام المتاحة). على سبيل المثال، قم بتحميل نموذج لتصنيف التسلسل باستخدام [`AutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

أعد استخدام نقطة التفتيش نفسها لتحميل بنية لمهمة مختلفة بسهولة:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip warning={true}>

بالنسبة لنماذج PyTorch، تستخدم طريقة `from_pretrained()` دالة `torch.load()` التي تستخدم داخليًا `pickle` والتي تُعرف بأنها غير آمنة. بشكل عام، لا تقم مطلقًا بتحميل نموذج قد يكون مصدره غير موثوق، أو قد يكون تم العبث به. يتم تخفيف هذا الخطر الأمني جزئيًا بالنسبة للنماذج العامة المستضافة على منصة Hugging Face Hub، والتي يتم [فحصها بحثًا عن البرامج الضارة](https://huggingface.co/docs/hub/security-malware) في كل مرة يتم فيها الالتزام. راجع [وثائق Hub](https://huggingface.co/docs/hub/security) للحصول على أفضل الممارسات مثل [التحقق من التوقيع](https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg) باستخدام GPG.

نماذج TensorFlow وFlax غير متأثرة، ويمكن تحميلها داخل بنى PyTorch باستخدام المعلمات `from_tf` و`from_flax` لطريقة `from_pretrained` لتجنب هذه المشكلة.

</Tip>

بشكل عام، نوصي باستخدام فئة `AutoTokenizer` وفئة `AutoModelFor` لتحميل مثيلات مُدربة مسبقًا من النماذج. سيساعدك هذا في تحميل البنية الصحيحة في كل مرة. في البرنامج التعليمي التالي، تعلم كيفية استخدام مُرمز الصور ومعالج الميزات والمعالج الذي تم تحميله حديثًا لمعالجة مجموعة بيانات للتدريب الدقيق.

</pt>
<tf>
أخيرًا، تتيح لك فئات `TFAutoModelFor` تحميل نموذج مُدرب مسبقًا لمهمة معينة (راجع [هنا](model_doc/auto) للحصول على قائمة كاملة بالمهام المتاحة). على سبيل المثال، قم بتحميل نموذج لتصنيف التسلسل باستخدام [`TFAutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

أعد استخدام نقطة التفتيش نفسها لتحميل بنية لمهمة مختلفة بسهولة:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

بشكل عام، نوصي باستخدام فئة `AutoTokenizer` وفئة `TFAutoModelFor` لتحميل مثيلات مُدربة مسبقًا من النماذج. سيساعدك هذا في تحميل البنية الصحيحة في كل مرة. في البرنامج التعليمي التالي، تعلم كيفية استخدام مُرمز الصور ومعالج الميزات والمعالج الذي تم تحميله حديثًا لمعالجة مجموعة بيانات للتدريب الدقيق.

</tf>
</frameworkcontent>