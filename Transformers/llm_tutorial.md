# التوليد باستخدام نماذج اللغة الضخمة

تعد نماذج اللغة الضخمة (LLMs) المكون الرئيسي وراء توليد النصوص. وباختصار، فهي تتكون من نماذج محولة كبيرة مسبقة التدريب تم تدريبها للتنبؤ بالكلمة التالية (أو، بشكل أكثر دقة، الرمز) بالنظر إلى نص الإدخال. نظرًا لأنها تتنبأ برمز واحد في كل مرة، يجب عليك القيام بشيء أكثر تفصيلاً لتوليد جمل جديدة بخلاف مجرد استدعاء النموذج - يجب عليك إجراء التوليد الانحداري الذاتي.

التوليد الانحداري الذاتي هو إجراء وقت الاستدلال الذي يستدعي النموذج بشكل تكراري مع المخرجات التي تم إنشاؤها الخاصة به، بالنظر إلى بعض الإدخالات الأولية. في مكتبة 🤗 Transformers، يتم التعامل مع هذا بواسطة طريقة [`~generation.GenerationMixin.generate`]، والتي تتوفر لجميع النماذج ذات القدرات التوليدية.

سيوضح لك هذا البرنامج التعليمي كيفية:

* توليد النص باستخدام نموذج اللغة الضخمة
* تجنب المشكلات الشائعة
* الخطوات التالية لمساعدتك في الاستفادة القصوى من نموذج اللغة الضخمة

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```

## توليد النص

يأخذ نموذج اللغة الذي تم تدريبه على [نمذجة اللغة السببية](tasks/language_modeling) تسلسل رموز النص كإدخال ويعيد توزيع الاحتمالية للرمز التالي.

يعد اختيار الرمز التالي من توزيع الاحتمالية هذا جانبًا بالغ الأهمية في التوليد الانحداري الذاتي باستخدام نماذج اللغة الضخمة. كل شيء مسموح به في هذه الخطوة طالما أنك تنتهي برمز للتكرار التالي. وهذا يعني أنه يمكن أن يكون بسيطًا مثل اختيار الرمز الأكثر احتمالًا من توزيع الاحتمالية أو معقدًا مثل تطبيق اثنتي عشرة تحويلًا قبل أخذ العينات من التوزيع الناتج.

تتكرر العملية الموضحة أعلاه بشكل تكراري حتى يتم الوصول إلى شرط التوقف. في الوضع المثالي، يملي النموذج شرط التوقف، والذي يجب أن يتعلم عند إخراج رمز نهاية التسلسل (`EOS`). إذا لم يكن الأمر كذلك، يتوقف التوليد عند الوصول إلى طول أقصى محدد مسبقًا.

من الضروري ضبط خطوة اختيار الرمز وشرط التوقف بشكل صحيح لجعل نموذجك يتصرف كما تتوقع في مهمتك. ولهذا السبب لدينا ملف [`~generation.GenerationConfig`] المرتبط بكل نموذج، والذي يحتوي على معلمة توليدية افتراضية جيدة ويتم تحميله جنبًا إلى جنب مع نموذجك.

دعنا نتحدث عن الكود!

<Tip>

إذا كنت مهتمًا بالاستخدام الأساسي لنموذج اللغة الضخمة، فإن واجهة [`Pipeline`](pipeline_tutorial) عالية المستوى لدينا هي نقطة بداية رائعة. ومع ذلك، غالبًا ما تتطلب نماذج اللغة الضخمة ميزات متقدمة مثل التكميم والتحكم الدقيق في خطوة اختيار الرمز، والتي يتم تنفيذها بشكل أفضل من خلال [`~generation.GenerationMixin.generate`]. يعد التوليد الانحداري الذاتي باستخدام نماذج اللغة الضخمة كثيف الاستخدام للموارد أيضًا، ويجب تنفيذه على وحدة معالجة الرسومات (GPU) لتحقيق الإنتاجية الكافية.

</Tip>

أولاً، تحتاج إلى تحميل النموذج.

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

ستلاحظ وجود علمين في مكالمة `from_pretrained`:

- `device_map` يضمن انتقال النموذج إلى وحدة (وحدات) معالجة الرسومات الخاصة بك
- `load_in_4bit` يطبق [التكميم الديناميكي 4-بت](main_classes/quantization) لتقليل متطلبات الموارد بشكل كبير

هناك طرق أخرى لتهيئة نموذج، ولكن هذا خط أساس جيد للبدء بنموذج اللغة الضخمة.

بعد ذلك، تحتاج إلى معالجة إدخال النص الخاص بك باستخدام محلل نحوي.

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

تحتوي متغير `model_inputs` على إدخال النص المعالج، بالإضافة إلى قناع الاهتمام. في حين أن [`~generation.GenerationMixin.generate`] تبذل قصارى جهدها لاستنتاج قناع الاهتمام عندما لا يتم تمريره، نوصي بتمريره كلما أمكن ذلك للحصول على نتائج مثالية.

بعد تحليل المدخلات، يمكنك استدعاء طريقة [`~generation.GenerationMixin.generate`] لإرجاع الرموز المولدة. يجب بعد ذلك تحويل الرموز المولدة إلى نص قبل الطباعة.

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, orange, purple, pink,'
```

أخيرًا، لا تحتاج إلى القيام بذلك تسلسل واحد في كل مرة! يمكنك تجميع إدخالاتك، والتي ستؤدي إلى تحسين الإنتاجية بشكل كبير بتكلفة صغيرة في الكمون والذاكرة. كل ما عليك فعله هو التأكد من إضافة علامات الترقيم إلى إدخالاتك بشكل صحيح (المزيد حول ذلك أدناه).

```py
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model_inputs = tokenizer(
...     ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['A list of colors: red, blue, green, yellow, orange, purple, pink,',
'Portugal is a country in southwestern Europe, on the Iber']
```

وهذا كل شيء! في بضع سطور من التعليمات البرمجية، يمكنك تسخير قوة نموذج اللغة الضخمة.

## المشكلات الشائعة

هناك العديد من [استراتيجيات التوليد](generation_strategies)، وفي بعض الأحيان قد لا تكون القيم الافتراضية مناسبة لحالتك الاستخدام. إذا لم تكن المخرجات الخاصة بك متوافقة مع ما تتوقعه، فقد قمنا بإنشاء قائمة بأكثر المشكلات شيوعًا وكيفية تجنبها.

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

### الإخراج المولد قصير جدًا/طويل جدًا

إذا لم يتم تحديده في ملف [`~generation.GenerationConfig`]`، فإن الدالة `generate` تعيد ما يصل إلى 20 رمزًا بشكل افتراضي. نوصي بشدة بتعيين `max_new_tokens` يدويًا في مكالمة `generate` للتحكم في العدد الأقصى من الرموز الجديدة التي يمكنها إرجاعها. ضع في اعتبارك أن نماذج اللغة الضخمة (بشكل أكثر دقة، [نماذج فك التشفير فقط](https://huggingface.co/learn/nlp-course/chapter1/6؟fw=pt)) تعيد أيضًا موجه الإدخال كجزء من الإخراج.

```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### وضع التوليد غير الصحيح

بشكل افتراضي، وما لم يتم تحديده في ملف [`~generation.GenerationConfig`]`، فإن الدالة `generate` تختار الرمز الأكثر احتمالًا في كل تكرار (فك تشفير جشع). اعتمادًا على مهمتك، قد يكون هذا غير مرغوب فيه؛ تستفيد المهام الإبداعية مثل الدردشة الآلية أو كتابة مقال من أخذ العينات. من ناحية أخرى، تستفيد المهام المستندة إلى الإدخال مثل نسخ النص الصوتي أو الترجمة من فك التشفير الجشع. قم بتمكين أخذ العينات باستخدام `do_sample=True`، ويمكنك معرفة المزيد حول هذا الموضوع في [منشور المدونة](https://huggingface.co/blog/how-to-generate) هذا.

```py
>>> # Set seed for reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(42)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.  Specifically, I am an indoor-only cat.  I'
```
### جانب الحشو الخاطئ

تعد نماذج اللغة الكبيرة (LLMs) بنى معمارية تعتمد على فك التشفير فقط، مما يعني أنها تواصل التكرار بناءً على موجه الإدخال الخاص بك. إذا لم يكن لإدخالاتك نفس الطول، فيجب حشوها. نظرًا لأن نماذج اللغة الكبيرة غير مدربة على الاستمرار من رموز الحشو، فيجب حشو إدخالها من اليسار. تأكد أيضًا من عدم نسيان تمرير قناع الاهتمام إلى وظيفة التوليد!

### موجه خاطئ

يتوقع بعض النماذج والمهام تنسيق موجه إدخال معين للعمل بشكل صحيح. عندما لا يتم تطبيق هذا التنسيق، ستحصل على تدهور صامت في الأداء: يعمل النموذج نوعًا ما، ولكن ليس كذلك إذا كنت تتبع الموجه المتوقع. تتوفر معلومات إضافية حول التوجيه، بما في ذلك النماذج والمهام التي تحتاج إلى توخي الحذر، في هذا الدليل. دعونا نرى مثالاً مع LLM للدردشة، والذي يستخدم قالب الدردشة:

## موارد إضافية

في حين أن عملية التوليد ذاتي الارتباط واضحة إلى حد ما، فإن تحقيق أقصى استفادة من نموذج اللغة الخاص بك يمكن أن يكون مهمة صعبة لأن هناك العديد من الأجزاء المتحركة. بالنسبة لخطواتك التالية لمساعدتك في الغوص بشكل أعمق في استخدام نموذج اللغة وفهمه:

### استخدام متقدم للتوليد

1. دليل حول كيفية التحكم في طرق التوليد المختلفة، وكيفية إعداد ملف تكوين التوليد، وكيفية بث الإخراج.
2. تسريع توليد النص.
3. قوالب موجهات لنماذج اللغة للدردشة.
4. دليل تصميم الموجه.
5. مرجع API على `~generation.GenerationConfig`، و`~generation.GenerationMixin.generate`، والصفوف المتعلقة بالتوليد. تحتوي معظم الفئات، بما في ذلك معالجات اللوغاريتم، على أمثلة للاستخدام!

### لوحات قيادة نموذج اللغة الكبيرة

1. لوحة قيادة Open LLM Leaderboard، والتي تركز على جودة النماذج مفتوحة المصدر.
2. لوحة قيادة Open LLM-Perf Leaderboard، والتي تركز على الإنتاجية نموذج اللغة الكبيرة.

### الكمون وسرعة المعالجة واستخدام الذاكرة

1. دليل حول كيفية تحسين نماذج اللغة الكبيرة للسرعة والذاكرة.
2. دليل حول التكميم، مثل bitsandbytes وautogptq، والذي يوضح لك كيفية تقليل متطلبات الذاكرة بشكل كبير.

### المكتبات ذات الصلة

1. "الأمثل"، وهو امتداد لـ "محولات" الذي يتم تحسينه لأجهزة الأجهزة المحددة.
2. "المخططات العامة"، وهي مكتبة يمكنك من خلالها تقييد توليد النص (على سبيل المثال، لتوليد ملفات JSON).
3. "text-generation-inference"، وهو خادم جاهز للإنتاج لنماذج اللغة الكبيرة.
4. "text-generation-webui"، وهو واجهة مستخدم لتوليد النص.