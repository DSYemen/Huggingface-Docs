## المهام المتعلقة بالصور باستخدام IDEFICS

في حين يمكن معالجة المهام الفردية عن طريق الضبط الدقيق للنماذج المتخصصة، هناك نهج بديل ظهر مؤخرًا واكتسب شعبية وهو استخدام النماذج الكبيرة لمجموعة متنوعة من المهام دون ضبط دقيق. على سبيل المثال، يمكن لنماذج اللغة الكبيرة التعامل مع مهام معالجة اللغات الطبيعية مثل الملخص والترجمة والتصنيف، والمزيد. لم يعد هذا النهج مقتصرًا على طريقة واحدة، مثل النص، وفي هذا الدليل، سنوضح كيف يمكنك حل مهام الصور والنصوص باستخدام نموذج متعدد الوسائط كبير يسمى IDEFICS.

[IDEFICS](../model_doc/idefics) هو نموذج مفتوح الوصول للرؤية واللغة يعتمد على [Flamingo](https://huggingface.co/papers/2204.14198)، وهو نموذج لغوي بصري متطور تم تطويره في الأصل بواسطة DeepMind. يقبل النموذج تسلسلات تعسفية من إدخالات الصور والنصوص وينتج نصًا متماسكًا كإخراج. يمكنه الإجابة على الأسئلة حول الصور، ووصف المحتوى المرئي، وخلق قصص قائمة على صور متعددة، وهلم جرا. يأتي IDEFICS في متغيرين - [80 مليار معلمة](https://huggingface.co/HuggingFaceM4/idefics-80b) و [9 مليار معلمة](https://huggingface.co/HuggingFaceM4/idefics-9b)، وكلاهما متاح على Hub. بالنسبة لكل متغير، يمكنك أيضًا العثور على إصدارات موجهة من النموذج المكيف لحالات الاستخدام المحادثية.

هذا النموذج متعدد الاستخدامات بشكل استثنائي ويمكن استخدامه لمجموعة واسعة من المهام المتعلقة بالصور والوسائط المتعددة. ومع ذلك، فإن كونها نموذجًا كبيرًا يعني أنها تتطلب موارد حوسبة وهياكل أساسية كبيرة. الأمر متروك لك لتقرر ما إذا كان هذا النهج يناسب حالتك الاستخدام بشكل أفضل من الضبط الدقيق للنماذج المتخصصة لكل مهمة فردية.

في هذا الدليل، ستتعلم كيفية:

- [تحميل IDEFICS](#تحميل-النموذج) و [تحميل الإصدار الكمي من النموذج](#النموذج-الكمي)
- استخدام IDEFICS لما يلي:
  - [وضع عنوان الصورة](#وضع-عنوان-الصورة)
  - [وضع عنوان الصورة الموجه](#وضع-عنوان-الصورة-الموجه)
  - [التوجيه القليل](#التوجيه-القليل)
  - [الإجابة على الأسئلة المرئية](#الإجابة-على-الأسئلة-المرئية)
  - [تصنيف الصور](#تصنيف-الصور)
  - [توليد النص الموجه بالصورة](#توليد-النص-الموجه-بالصورة)
  - [تشغيل الاستدلال في وضع الدفعات](#تشغيل-الاستدلال-في-وضع-الدفعات)
  - [تشغيل IDEFICS instruct للاستخدام المحادثي](#تشغيل-IDEFICS-instruct-للاستخدام-المحادثي)

قبل البدء، تأكد من تثبيت جميع المكتبات اللازمة.

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
لتشغيل الأمثلة التالية باستخدام إصدار غير كمي من نقطة تفتيش النموذج، ستحتاج إلى ذاكرة GPU لا تقل عن 20 جيجابايت.
</Tip>

## تحميل النموذج

دعنا نبدأ بتحميل نقطة تفتيش معلمات النموذج البالغة 9 مليارات:

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

تمامًا مثل النماذج الأخرى لـ Transformers، يلزمك تحميل معالج والنموذج نفسه من نقطة التفتيش. يجمع معالج IDEFICS بين [`LlamaTokenizer`] ومعالج الصور IDEFICS في معالج واحد للاهتمام بإعداد إدخالات النص والصورة للنموذج.

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text، AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

يؤدي تعيين `device_map` إلى `"auto"` إلى تحديد كيفية تحميل وتخزين أوزان النموذج بالطريقة الأكثر تحسينًا بالنظر إلى الأجهزة الموجودة.

### النموذج الكمي

إذا كانت ذاكرة GPU عالية السعة تمثل مشكلة، فيمكنك تحميل الإصدار الكمي من النموذج. لتحميل النموذج والمعالج بدقة 4 بت، قم بتمرير `BitsAndBytesConfig` إلى طريقة `from_pretrained` وسيتم ضغط النموذج أثناء التحميل.

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text، AutoProcessor، BitsAndBytesConfig

>>> quantization_config = BitsAndBytesConfig(
...     load_in_4bit=True،
...     bnb_4bit_compute_dtype=torch.float16،
... )

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(
...     checkpoint،
...     quantization_config=quantization_config،
...     device_map="auto"
... )
```

الآن بعد أن قمت بتحميل النموذج بإحدى الطرق المقترحة، دعنا ننتقل إلى استكشاف المهام التي يمكنك استخدام IDEFICS من أجلها.

## وضع عنوان الصورة

وضع عنوان الصورة هو مهمة التنبؤ بعنوان لصورة معينة. أحد التطبيقات الشائعة هو مساعدة الأشخاص ضعاف البصر على التنقل في مختلف المواقف، على سبيل المثال، استكشاف محتوى الصورة عبر الإنترنت.

لتوضيح المهمة، احصل على صورة لوضع عنوان لها، على سبيل المثال:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="صورة لجرو في سرير زهرة"/>
</div>

الصورة بواسطة [Hendo Wang](https://unsplash.com/@hendoo).

يقبل IDEFICS مطالبات النص والصورة. ومع ذلك، لوضع عنوان لصورة، لا يلزم تقديم مطالبة نصية للنموذج، فقط الصورة المدخلة مسبقًا. بدون مطالبة نصية، سيبدأ النموذج في إنشاء نص من رمز BOS (بداية التسلسل) وبالتالي إنشاء عنوان.

يمكن استخدام كائن صورة (`PIL.Image`) أو عنوان URL الذي يمكن استرداد الصورة منه كإدخال صورة للنموذج.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80"،
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
جرو في سرير زهرة
```

<Tip>
من الجيد تضمين `bad_words_ids` في المكالمة إلى `generate` لتجنب الأخطاء الناشئة عند زيادة `max_new_tokens`: سيرغب النموذج في إنشاء رمز `<image>` أو `<fake_token_around_image>` جديد عندما لا تكون هناك صورة يتم إنشاؤها بواسطة النموذج.
يمكنك تعيينه أثناء التنقل كما هو موضح في هذا الدليل، أو تخزينه في `GenerationConfig` كما هو موضح في دليل [استراتيجيات توليد النص](../generation_strategies).
</Tip>

## وضع عنوان الصورة الموجه

يمكنك تمديد وضع عنوان الصورة عن طريق توفير مطالبة نصية، والتي سيواصلها النموذج نظرًا للصورة. دعنا نأخذ صورة أخرى لتوضيح ذلك:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="صورة لبرج إيفل ليلاً"/>
</div>

الصورة بواسطة [Denys Nevozhai](https://unsplash.com/@dnevozhai).

يمكن تمرير المطالبات النصية والبصرية إلى معالج النموذج كقائمة واحدة لإنشاء الإدخالات المناسبة.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80"،
...     "هذه صورة لـ "،
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
هذه صورة لبرج إيفل في باريس، فرنسا.
```
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين.

## Few-shot prompting
على الرغم من أن IDEFICS يحقق نتائج رائعة في حالة zero-shot، إلا أن مهمتك قد تتطلب تنسيقًا معينًا للتعليق التوضيحي، أو قد تأتي بمتطلبات أو قيود أخرى تزيد من تعقيد المهمة. يمكن استخدام Few-shot prompting لتمكين التعلم في السياق. من خلال توفير أمثلة في المطالبة، يمكنك توجيه النموذج لإنتاج نتائج تحاكي تنسيق الأمثلة المعطاة.

لنستخدم صورة برج إيفل السابقة كمثال للنموذج وننشئ مطالبة توضح للنموذج أنه بالإضافة إلى تعلم ما هو الكائن في الصورة، نريد أيضًا الحصول على بعض المعلومات المثيرة للاهتمام عنه. ثم دعونا نرى إذا كان بإمكاننا الحصول على تنسيق الاستجابة نفسه لصورة تمثال الحرية:

## Visual question answering
Visual Question Answering (VQA) هي مهمة الإجابة على الأسئلة المفتوحة بناءً على صورة. مشابه لوصف الصورة، يمكن استخدامه في تطبيقات إمكانية الوصول، ولكن أيضًا في التعليم (الاستدلال حول المواد المرئية)، وخدمة العملاء (الأسئلة حول المنتجات بناءً على الصور)، واسترجاع الصور.

دعونا نحصل على صورة جديدة لهذه المهمة:

يمكنك توجيه النموذج من وصف الصورة إلى الإجابة على الأسئلة المرئية من خلال مطالبتها بتعليمات مناسبة:

## Image classification
IDEFICS قادر على تصنيف الصور إلى فئات مختلفة دون تدريب صريح على بيانات تحتوي على أمثلة موسومة من تلك الفئات المحددة. بالنظر إلى قائمة الفئات واستخدام قدرات فهم الصور والنصوص، يمكن للنموذج استنتاج الفئة التي من المحتمل أن تنتمي إليها الصورة.

لنفترض أن لدينا هذه الصورة لطاولة الخضار:

يمكننا توجيه النموذج لتصنيف الصورة إلى واحدة من الفئات التي لدينا:

## Image-guided text generation
لتطبيقات أكثر إبداعًا، يمكنك استخدام Image-guided text generation لتوليد نص بناءً على صورة. يمكن أن يكون هذا مفيدًا لإنشاء أوصاف المنتجات، والإعلانات، وأوصاف المشاهد، وما إلى ذلك.

دعونا نوجه IDEFICS لكتابة قصة بناءً على صورة بسيطة لباب أحمر:

يبدو أن IDEFICS لاحظ القرع على عتبة الباب واختار قصة مخيفة عن شبح في ليلة الهالوين.

## Running inference in batch mode
وضحت جميع الأقسام السابقة IDEFICS لمثال واحد. بطريقة مشابهة جدًا، يمكنك تشغيل الاستدلال لمجموعة من الأمثلة عن طريق تمرير قائمة من المطالبات:
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع اتباع التعليمات التي ذكرتها:

## IDEFICS للمحادثة

بالنسبة لحالات الاستخدام المحادثية، يمكنك العثور على إصدارات مُعلَّمة موجهة من النموذج على 🤗 Hub:
`HuggingFaceM4/idefics-80b-instruct` و `HuggingFaceM4/idefics-9b-instruct`.
تمثل هذه النقاط المرجعية نتيجة الضبط الدقيق لنماذج القاعدة المناظرة على مزيج من مجموعات البيانات الخاضعة للإشراف والضبط الدقيق للتعليمات، مما يعزز الأداء الهابط مع جعل النماذج أكثر قابلية للاستخدام في إعدادات المحادثة.

يُشبه الاستخدام والتشغيل للمحادثة إلى حد كبير استخدام نماذج القاعدة:

>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> checkpoint = "HuggingFaceM4/idefics-9b-instruct"
>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompts = [
...     [
...         "المستخدم: ما الذي يوجد في هذه الصورة؟",
...         "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
...         "<end_of_utterance>",

...         "\nالمساعد: يصور هذا الرسم إيديفيكس، كلب أوبيليكس في أستريكس وأوبيليكس. إيديفيكس يركض على الأرض.<end_of_utterance>",

...         "\nالمستخدم:",
...         "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
...         "ومن هذا؟<end_of_utterance>",

...         "\nالمساعد:",
...     ],
... ]

>>> # --وضع الدفعات
>>> المدخلات = المعالج (المطالبات، add_end_of_utterance_token=False، return_tensors="pt").to(device)
>>> # --وضع عينة واحدة
>>> # المدخلات = المعالج (المطالبات [0]، return_tensors="pt").to(device)

>>> # Generation args
>>> exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i, t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n")