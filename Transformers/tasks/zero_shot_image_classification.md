# التصنيف الصوري الصفري 

التصنيف الصوري الصفري هي مهمة تتضمن تصنيف الصور إلى فئات مختلفة باستخدام نموذج لم يتم تدريبه بشكل صريح على بيانات تحتوي على أمثلة موسومة من تلك الفئات المحددة.

تتطلب تصنيف الصور التقليدية تدريب نموذج على مجموعة محددة من الصور الموسومة، ويتعلم هذا النموذج "رسم خريطة" لميزات الصورة إلى التسميات. عندما تكون هناك حاجة لاستخدام هذا النموذج لمهمة تصنيف تقدم مجموعة جديدة من التسميات، يلزم الضبط الدقيق "لإعادة معايرة" النموذج.

على النقيض من ذلك، فإن نماذج التصنيف الصوري الصفري أو ذات المفردات المفتوحة هي نماذج متعددة الوسائط تم تدريبها عادةً على مجموعة كبيرة من الصور ووصفاتها المرتبطة. تتعلم هذه النماذج تمثيلات الرؤية واللغة المحاذاة التي يمكن استخدامها للعديد من المهام النهائية بما في ذلك التصنيف الصوري الصفري.

هذا نهج أكثر مرونة لتصنيف الصور يسمح للنماذج بالتعميم على فئات جديدة وغير مرئية دون الحاجة إلى بيانات تدريب إضافية، ويمكّن المستخدمين من استعلام الصور بوصف نصي حر للأشياء المستهدفة.

في هذا الدليل، ستتعلم كيفية:

- إنشاء خط أنابيب التصنيف الصوري الصفري
- تشغيل استنتاج التصنيف الصوري الصفري يدويًا

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install -q "transformers[torch]" pillow
```

## خط أنابيب التصنيف الصوري الصفري

أبسط طريقة لتجربة الاستنتاج باستخدام نموذج يدعم التصنيف الصوري الصفري هي استخدام خط الأنابيب المقابل. قم بتنفيذ خط أنابيب من [نقطة تفتيش على Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads):

```python
>>> from transformers import pipeline

>>> checkpoint = "openai/clip-vit-large-patch14"
>>> detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
```

بعد ذلك، اختر صورة تريد تصنيفها.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/owl.jpg" alt="صورة بومة"/>
</div>

مرر الصورة وتسميات الكائنات المرشحة إلى خط الأنابيب. هنا نمرر الصورة مباشرةً؛ تشمل الخيارات المناسبة الأخرى مسارًا محليًا إلى صورة أو عنوان URL للصورة.

يمكن أن تكون التسميات المرشحة كلمات بسيطة كما هو موضح في هذا المثال، أو أكثر وصفًا.

```py
>>> predictions = detector(image, candidate_labels=["fox", "bear", "seagull", "owl"])
>>> predictions
[{'score': 0.9996670484542847, 'label': 'owl'},
{'score': 0.000199399160919711, 'label': 'seagull'},
{'score': 7.392891711788252e-05, 'label': 'fox'},
{'score': 5.96074532950297e-05, 'label': 'bear'}]
```

## التصنيف الصوري الصفري يدويًا

الآن بعد أن رأيت كيفية استخدام خط أنابيب التصنيف الصوري الصفري، دعنا نلقي نظرة على كيفية تشغيل التصنيف الصوري الصفري يدويًا.

ابدأ بتحميل النموذج والمعالج المرتبط من [نقطة تفتيش على Hugging Face Hub](https://huggingface.co/models؟pipeline_tag=zero-shot-image-classification&sort=downloads).

سنستخدم هنا نفس نقطة التفتيش كما هو موضح سابقًا:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

>>> model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

دعنا نأخذ صورة مختلفة للتغيير.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" alt="صورة سيارة"/>
</div>

استخدم المعالج لتحضير المدخلات للنموذج. يجمع المعالج بين معالج الصور الذي يقوم بإعداد الصورة للنموذج عن طريق تغيير حجمها وتطبيعها، ومعالج نصوص يهتم بالمدخلات النصية.

```py
>>> candidate_labels = ["tree", "car", "bike", "cat"]
>>> inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
```

مرر المدخلات عبر النموذج، وقم بمعالجة النتائج:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits_per_image[0]
>>> probs = logits.softmax(dim=-1).numpy()
>>> scores = probs.tolist()

>>> result = [
...     {"score": score, "label": candidate_label}
...     for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
... ]

>>> result
[{'score': 0.998572, 'label': 'car'},
{'score': 0.0010570387, 'label': 'bike'},
{'score': 0.0003393686, 'label': 'tree'},
{'score': 3.1572064e-05, 'label': 'cat'}]
```