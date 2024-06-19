# خطوط الأنابيب للاستنتاج

تجعل [`pipeline`] من السهل استخدام أي نموذج من [Hub](https://huggingface.co/models) للاستنتاج لأي مهام خاصة باللغة أو الرؤية الحاسوبية أو الكلام أو متعددة الوسائط. حتى إذا لم يكن لديك خبرة في طريقة معينة أو لم تكن على دراية بالرمز الأساسي وراء النماذج، فيمكنك لا تزال استخدامها للاستنتاج مع [`pipeline`]! سيعلمك هذا البرنامج التعليمي ما يلي:

* استخدام [`pipeline`] للاستنتاج.
* استخدام محلل لغوي أو نموذج محدد.
* استخدام [`pipeline`] للمهام الصوتية والبصرية ومتعددة الوسائط.

<Tip>

الق نظرة على وثائق [`pipeline`] للحصول على قائمة كاملة بالمهام المدعومة والمعلمات المتاحة.

</Tip>

## استخدام خط الأنابيب

على الرغم من أن لكل مهمة خط أنابيب [`pipeline`] مرتبط بها، إلا أنه من الأبسط استخدام التجريد العام لخط الأنابيب [`pipeline`] الذي يحتوي على جميع خطوط الأنابيب الخاصة بالمهمة. يقوم خط الأنابيب [`pipeline`] تلقائيًا بتحميل نموذج افتراضي وفئة معالجة مسبقة قادرة على الاستدلال لمهمتك. دعنا نأخذ مثال استخدام خط الأنابيب للاستعراف الآلي للكلام (ASR)، أو تحويل الكلام إلى نص.

1. ابدأ بإنشاء خط أنابيب [`pipeline`] وحدد مهمة الاستدلال:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition")
```

2. مرر إدخالك إلى خط الأنابيب. في حالة التعرف على الكلام، يكون هذا ملف إدخال صوتي:

```py
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

هل هذه هي النتيجة التي كنت تبحث عنها؟ تحقق من بعض [نماذج التعرف على الكلام الأكثر تنزيلًا](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending) على Hub لمعرفة ما إذا كان يمكنك الحصول على نسخة منقحة أفضل.

دعنا نجرب نموذج [Whisper large-v2](https://huggingface.co/openai/whisper-large) من OpenAI. تم إصدار Whisper بعد عامين من إصدار Wav2Vec2، وتم تدريبه على ما يقرب من 10 أضعاف كمية البيانات. وبهذه الطريقة، فإنه يتفوق على Wav2Vec2 في معظم المعايير المرجعية لأسفل البئر. كما أن لديها الفائدة الإضافية المتمثلة في التنبؤ بعلامات الترقيم وعلامات الحالة، والتي لا يمكن القيام بها مع Wav2Vec2.

دعنا نجربها هنا لنرى كيف تؤدي:

```py
>>> transcriber = pipeline(model="openai/whisper-large-v2")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

الآن تبدو هذه النتيجة أكثر دقة! للحصول على مقارنة متعمقة بين Wav2Vec2 مقابل Whisper، راجع [دورة Audio Transformers](https://huggingface.co/learn/audio-course/chapter5/asr_models).

نحن نشجعك حقًا على التحقق من Hub للحصول على نماذج بلغات مختلفة، ونماذج متخصصة في مجالك، والمزيد.

يمكنك التحقق من نتائج النموذج ومقارنتها مباشرة من متصفحك على Hub لمعرفة ما إذا كان يناسبها أو يتعامل مع الحالات الحدودية بشكل أفضل من النماذج الأخرى.

وإذا لم تجد نموذجًا لحالتك الاستخدامية، فيمكنك دائمًا البدء في [تدريب](training) نموذجك الخاص!

إذا كان لديك عدة مدخلات، فيمكنك تمرير إدخالك كقائمة:

```py
transcriber(
[
"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
]
)
```

تعد خطوط الأنابيب رائعة للتجريب نظرًا لأن التبديل من نموذج إلى آخر أمر بسيط؛ ومع ذلك، هناك بعض الطرق لتحسينها لأحمال العمل الأكبر من التجريب. راجع الأدلة التالية التي تتعمق في التكرار عبر مجموعات البيانات بأكملها أو استخدام خطوط الأنابيب في خادم ويب:

* [استخدام خطوط الأنابيب على مجموعة بيانات](#using-pipelines-on-a-dataset)
* [استخدام خطوط الأنابيب لخادم ويب](./pipeline_webserver)

## المعلمات

تدعم [`pipeline`] العديد من المعلمات؛ بعضها خاص بالمهمة، والبعض الآخر عام لجميع خطوط الأنابيب.

بشكل عام، يمكنك تحديد المعلمات في أي مكان تريده:

```py
transcriber = pipeline(model="openai/whisper-large-v2", my_parameter=1)

out = transcriber(...)  # سيتم استخدام هذا `my_parameter=1`.
out = transcriber(..., my_parameter=2)  # سيتم تجاوز هذا واستخدام `my_parameter=2`.
out = transcriber(...)  # سيتم الرجوع إلى استخدام `my_parameter=1`.
```

دعنا نلقي نظرة على 3 مهمة:

### الجهاز

إذا كنت تستخدم `device=n`، فإن خط الأنابيب يضع النموذج تلقائيًا على الجهاز المحدد.

سيعمل هذا بغض النظر عما إذا كنت تستخدم PyTorch أو Tensorflow.

```py
transcriber = pipeline(model="openai/whisper-large-v2", device=0)
```

إذا كان النموذج كبيرًا جدًا بالنسبة لوحدة معالجة الرسومات (GPU) واحدة، وأنت تستخدم PyTorch، فيمكنك تعيين `device_map="auto"` لتحديد كيفية تحميل مخازن الأوزان النموذجية وتخزينها تلقائيًا. يتطلب استخدام حجة `device_map` حزمة 🤗 [Accelerate](https://huggingface.co/docs/accelerate):

```bash
pip install --upgrade accelerate
```

يقوم الرمز التالي بتحميل مخازن أوزان النموذج وتخزينها تلقائيًا عبر الأجهزة:

```py
transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")
```

لاحظ أنه إذا تم تمرير `device_map="auto"`، فلا توجد حاجة لإضافة حجة `device=device` عند إنشاء مثيل خط الأنابيب الخاص بك، وإلا فقد تواجه بعض السلوكيات غير المتوقعة!

### حجم الدفعة

بشكل افتراضي، لن تقوم خطوط الأنابيب بالدفعات للاستدلال لأسباب موضحة بالتفصيل [هنا](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching). والسبب هو أن الدفعات ليست أسرع بالضرورة، ويمكن أن تكون في الواقع أبطأ بكثير في بعض الحالات.

ولكن إذا نجحت في حالتك الاستخدامية، فيمكنك استخدام ما يلي:

```py
transcriber = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)
```

هذا يشغل خط الأنابيب على ملفات الصوت الأربعة المقدمة، ولكنه سيمررها في دفعات من 2
إلى النموذج (الذي يوجد على وحدة معالجة الرسومات (GPU)، حيث من المرجح أن تساعد الدفعات) دون الحاجة إلى أي رمز إضافي منك.

يجب أن تتطابق الإخراج دائمًا مع ما كنت ستحصل عليه بدون الدفعات. المقصود منه فقط كطريقة لمساعدتك في الحصول على المزيد من السرعة من خط الأنابيب.

يمكن لخطوط الأنابيب أيضًا تخفيف بعض التعقيدات في الدفعات لأنه، بالنسبة لبعض خطوط الأنابيب، يجب تقسيم عنصر واحد (مثل ملف صوتي طويل) إلى أجزاء متعددة لمعالجتها بواسطة نموذج. يقوم خط الأنابيب بأداء هذا [*chunk batching*](./main_classes/pipelines#pipeline-chunk-batching) نيابة عنك.

### معلمات خاصة بالمهمة

توفر جميع المهام معلمات خاصة بالمهمة تتيح المرونة والخيارات الإضافية لمساعدتك في أداء عملك.

على سبيل المثال، تحتوي طريقة [`transformers.AutomaticSpeechRecognitionPipeline.__call__`] على معلمة `return_timestamps` التي تبدو واعدة لترجمة مقاطع الفيديو:

```py
>>> transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.', 'chunks': [{'timestamp': (0.0, 11.88), 'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its'}, {'timestamp': (11.88, 12.38), 'text': ' creed.'}]}
```

كما ترون، استنتج النموذج النص وأخرج أيضًا **متى** تم نطق الجمل المختلفة.

تتوفر العديد من المعلمات لكل مهمة، لذا تحقق من مرجع API لكل مهمة لمعرفة ما يمكنك العبث به!

على سبيل المثال، تحتوي [`~transformers.AutomaticSpeechRecognitionPipeline`] على معلمة `chunk_length_s` التي تكون مفيدة
للعمل على ملفات الصوت الطويلة جدًا (على سبيل المثال، ترجمة الأفلام أو مقاطع الفيديو التي مدتها ساعة) والتي لا يمكن للنموذج عادةً
التعامل معها بمفردها:

```python
>>> transcriber = pipeline(model="openai/whisper-large-v2", chunk_length_s=30)
>>> transcriber("https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav")
{'text': " So in college, I was a government major, which means I had to write a lot of papers. Now, when a normal student writes a paper, they might spread the work out a little like this. So, you know. You get started maybe a little slowly, but you get enough done in the first week that with some heavier days later on, everything gets done and things stay civil. And I would want to do that like that. That would be the plan. I would have it all ready to go, but then actually the paper would come along, and then I would kind of do this. And that would happen every single paper. But then came my 90-page senior thesis, a paper you're supposed to spend a year on. I knew for a paper like that, my normal workflow was not an option, it was way too big a project. So I planned things out and I decided I kind of had to go something like this. This is how the year would go. So I'd start off light and I'd bump it up"}
```

إذا لم تتمكن من العثور على معلمة قد تساعدك حقًا، فلا تتردد في [طلبها](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml)!
## استخدام خطوط الأنابيب على مجموعة بيانات

يمكن لخط الأنابيب أيضًا إجراء الاستدلال على مجموعة بيانات كبيرة. أسهل طريقة نوصي بها للقيام بذلك هي باستخدام مؤشر:

تعطي الدالة iterator `data()` كل نتيجة، ويتم التعرف على خط الأنابيب تلقائيًا على أن الإدخال قابل للتنقل وسيبدأ في جلب البيانات أثناء استمراره في معالجتها على وحدة معالجة الرسومات (GPU) (يستخدم هذا [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) تحت الغطاء). هذا أمر مهم لأنك لا تحتاج إلى تخصيص ذاكرة لمجموعة البيانات بأكملها، ويمكنك إطعام وحدة معالجة الرسومات (GPU) بأسرع ما يمكن.

نظرًا لأن المعالجة الدُفعية يمكن أن تسرع الأمور، فقد يكون من المفيد محاولة ضبط معلمة "حجم الدُفعة" هنا.

أبسط طريقة للتنقل خلال مجموعة من البيانات هي تحميل واحدة من 🤗 [Datasets](https://github.com/huggingface/datasets/):

## استخدام خطوط الأنابيب لخادم ويب

<Tip>

إن إنشاء محرك استدلال هو موضوع معقد يستحق صفحته الخاصة.

[رابط](./pipeline_webserver)

## خط أنابيب الرؤية

إن استخدام خط أنابيب [`pipeline`] لمهمة الرؤية مطابق تقريبًا.

حدد مهمتك ومرر صورتك إلى المصنف. يمكن أن تكون الصورة رابطًا أو مسارًا محليًا أو صورة مشفرة بتنسيق Base64. على سبيل المثال، ما نوع فصيلة القط الموضح أدناه؟

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

## خط أنابيب النص

إن استخدام خط أنابيب [`pipeline`] لمهمة معالجة اللغات الطبيعية (NLP) مطابق تقريبًا.

## خط أنابيب متعدد الوسائط

يدعم خط الأنابيب [`pipeline`] أكثر من وسيط واحد. على سبيل المثال، تجمع مهمة الإجابة على الأسئلة البصرية (VQA) بين النص والصورة. لا تتردد في استخدام أي رابط صورة تريده وسؤال تريد طرحه حول الصورة. يمكن أن تكون الصورة عنوان URL أو مسارًا محليًا إلى الصورة.

على سبيل المثال، إذا كنت تستخدم هذه [صورة الفاتورة](https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png):

## استخدام `pipeline` على نماذج كبيرة مع 🤗 `accelerate`:

يمكنك تشغيل `pipeline` بسهولة على نماذج كبيرة باستخدام 🤗 `accelerate`! أولاً، تأكد من تثبيت `accelerate` باستخدام `pip install accelerate`.

قم أولاً بتحميل نموذجك باستخدام `device_map="auto"`! سنستخدم `facebook/opt-1.3b` كمثال لنا.

يمكنك أيضًا تمرير نماذج 8 بت إذا قمت بتثبيت `bitsandbytes` وإضافة الحجة `load_in_8bit=True`

ملاحظة: يمكنك استبدال نقطة التفتيش بأي نموذج من نماذج Hugging Face التي تدعم تحميل النماذج الكبيرة، مثل BLOOM.

## إنشاء عروض توضيحية ويب من خطوط الأنابيب باستخدام `gradio`

يتم دعم خطوط الأنابيب تلقائيًا في [Gradio](https://github.com/gradio-app/gradio/)، وهي مكتبة تجعل إنشاء تطبيقات تعليم الآلة الجميلة وسهلة الاستخدام على الويب نسيمًا. أولاً، تأكد من تثبيت Gradio:

ثم، يمكنك إنشاء عرض توضيحي ويب حول خط أنابيب تصنيف الصور (أو أي خط أنابيب آخر) في سطر واحد من التعليمات البرمجية عن طريق استدعاء وظيفة Gradio's [`Interface.from_pipeline`](https://www.gradio.app/docs/interface#interface-from-pipeline) لإطلاق خط الأنابيب. يقوم هذا بإنشاء واجهة بديهية للسحب والإفلات في مستعرضك:

بشكل افتراضي، يعمل العرض التوضيحي للويب على خادم محلي. إذا كنت ترغب في مشاركته مع الآخرين، فيمكنك إنشاء رابط عام مؤقت عن طريق تعيين `share=True` في `launch()`. يمكنك أيضًا استضافة عرضك التوضيحي على [Hugging Face Spaces](https://huggingface.co/spaces) للحصول على رابط دائم.