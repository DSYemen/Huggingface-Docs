بالتأكيد، سألتزم بالتعليمات المذكورة في طلب الترجمة. فيما يلي ترجمة النص الموجود في الفقرات والعناوين:

# خطوط أنابيب Marigold لمهام رؤية الكمبيوتر

Marigold هي طريقة جديدة للتنبؤ الكثيف القائم على الانتشار، ومجموعة من خطوط الأنابيب لمهام رؤية الكمبيوتر المختلفة، مثل تقدير العمق الأحادي. سيُظهر لك هذا الدليل كيفية استخدام Marigold للحصول على تنبؤات سريعة وعالية الجودة للصور ومقاطع الفيديو.

يدعم كل خط أنابيب مهمة رؤية حاسوبية واحدة، والتي تتخذ كإدخال صورة RGB وتنتج *تنبؤًا* بالنمط الذي يهمك، مثل خريطة العمق للصورة المدخلة.

حاليًا، يتم تنفيذ المهام التالية:

| خط الأنابيب | الأنماط المتوقعة | العروض التوضيحية |
| ------------ | ----------------- | ----------------- |
| MarigoldDepthPipeline | العمق، التباين | العرض التوضيحي السريع (LCM)، العرض التوضيحي الأصلي البطيء (DDIM) |
| MarigoldNormalsPipeline | القواعد السطحية العادية | العرض التوضيحي السريع (LCM) |

يمكن العثور على نقاط التفتيش الأصلية في منظمة [PRS-ETH](https://huggingface.co/prs-eth/) Hugging Face.

يُقصد بنقاط التفتيش هذه العمل مع خطوط أنابيب الناشرات و [قاعدة التعليمات البرمجية الأصلية](https://github.com/prs-eth/marigold). يمكن أيضًا استخدام التعليمات البرمجية الأصلية لتدريب نقاط تفتيش جديدة.

| نقطة التفتيش | النمط | التعليق |
| ------------ | ----- | -------- |
| prs-eth/marigold-v1-0 | العمق | أول نقطة تفتيش لعمق Marigold، والتي تتنبأ بخرائط العمق *الدقيقة للتحويلات المائلة*. تمت دراسة أداء نقطة التفتيش هذه في المعايير في الورقة [الأصلية](https://huggingface.co/papers/2312.02145). تم تصميمه ليتم استخدامه مع `DDIMScheduler` أثناء الاستدلال، فهو يتطلب ما لا يقل عن 10 خطوات للحصول على تنبؤات موثوقة. يتراوح التنبؤ بالعمق غير المائل بين القيم في كل بكسل بين 0 (الطائرة القريبة) و 1 (الطائرة البعيدة)؛ تختار كل من الطائرتين النموذج كجزء من عملية الاستدلال. راجع مرجع `MarigoldImageProcessor` للحصول على برامج مساعدة للتصور. |
| prs-eth/marigold-depth-lcm-v1-0 | العمق | نقطة تفتيش Marigold Depth السريعة، والتي تمت معايرتها الدقيقة من `prs-eth/marigold-v1-0`. تم تصميمه ليتم استخدامه مع `LCMScheduler` أثناء الاستدلال، فهو يتطلب خطوة واحدة فقط للحصول على تنبؤات موثوقة. تصل موثوقية التنبؤ إلى التشبع عند 4 خطوات وتنخفض بعد ذلك. |
| prs-eth/marigold-normals-v0-1 | القواعد | نقطة تفتيش معاينة لخط أنابيب Marigold Normals. تم تصميمه ليتم استخدامه مع `DDIMScheduler` أثناء الاستدلال، فهو يتطلب ما لا يقل عن 10 خطوات للحصول على تنبؤات موثوقة. تنبؤات القواعد السطحية هي متجهات 3D ذات طول وحدة بقيم تتراوح من -1 إلى 1. *سيتم إيقاف هذه النقطة بعد إصدار إصدار `v1-0`.* |
| prs-eth/marigold-normals-lcm-v0-1 | القواعد | نقطة تفتيش Marigold Normals السريعة، والتي تمت معايرتها الدقيقة من `prs-eth/marigold-normals-v0-1`. تم تصميمه ليتم استخدامه مع `LCMScheduler` أثناء الاستدلال، فهو يتطلب خطوة واحدة فقط للحصول على تنبؤات موثوقة. تصل موثوقية التنبؤ إلى التشبع عند 4 خطوات وتنخفض بعد ذلك. *سيتم إيقاف هذه النقطة بعد إصدار إصدار `v1-0`.* |

تُعطى الأمثلة أدناه في الغالب للتنبؤ بالعمق، ولكن يمكن تطبيقها عالميًا مع الأنماط الأخرى المدعومة. نحن نعرض التنبؤات باستخدام نفس صورة المدخلات لألبرت أينشتاين التي تم إنشاؤها بواسطة Midjourney.

يجعل هذا من السهل مقارنة تصورات التنبؤات عبر مختلف الأنماط ونقاط التفتيش.

### دليل سريع للتنبؤ بالعمق

للحصول على أول تنبؤ بالعمق، قم بتحميل نقطة تفتيش `prs-eth/marigold-depth-lcm-v1-0` في خط أنابيب `MarigoldDepthPipeline`، ومرر الصورة عبر خط الأنابيب، واحفظ التنبؤات:

ستجد أدناه التنبؤ الخام والمرئي؛ كما يمكنك أن ترى، من الأسهل تمييز المناطق الداكنة (الشارب) في التصور:

هل يمكنني تقديم أي مساعدة أخرى مع الترجمة؟
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين.

### بدء التنبؤ بسرعة السطح

قم بتحميل نقطة تفتيش "prs-eth/marigold-normals-lcm-v0-1" في خط أنابيب "MarigoldNormalsPipeline"، ومرر الصورة عبر خط الأنابيب، واحفظ التنبؤات:

توفر دالة التصور للمتجهات العادية ['~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_normals`] خريطة للتنبؤ ثلاثي الأبعاد مع قيم بكسل في النطاق '[-1، 1]' إلى صورة RGB.

تدعم دالة التصور عكس محاور المتجهات العادية للسطح لجعل التصور متوافقًا مع خيارات أخرى لإطار المرجع.

مفهوميًا، يتم طلاء كل بكسل وفقًا لمتجه العادي للسطح في إطار المرجع، حيث يشير المحور 'X' إلى اليمين، ويشير المحور 'Y' إلى الأعلى، ويشير المحور 'Z' إلى المشاهد.

فيما يلي التنبؤ المرئي:

في هذا المثال، من المؤكد أن طرف الأنف لديه نقطة على السطح، حيث يشير متجه العادي للسطح مباشرة إلى المشاهد، مما يعني أن إحداثياته هي '[0، 0، 1]'.

يتم رسم هذا المتجه إلى RGB '[128، 128، 255]'، والذي يقابل اللون الأزرق البنفسجي.

بالمثل، فإن المتجه العادي للسطح على الخد في الجزء الأيمن من الصورة لديه مكون 'X' كبير، والذي يزيد من اللون الأحمر.

تعزز النقاط الموجودة على الكتفين والتي تشير إلى الأعلى مع مكون 'Y' كبير اللون الأخضر.

### تسريع الاستنتاج

تم بالفعل تحسين مقتطفات بدء التشغيل السريع أعلاه للسرعة: فهي تحمل نقطة تفتيش LCM، وتستخدم متغير 'fp16' للأوزان والحساب، وتؤدي خطوة إزالة التشويش فقط.

يستغرق استدعاء 'pipe(image)` 280 مللي ثانية للاكتمال على وحدة معالجة الرسومات RTX 3090.

داخلياً، يتم ترميز الصورة المدخلة باستخدام مشفر VAE من Stable Diffusion، ثم يقوم U-Net بتنفيذ خطوة إزالة التشويش، وأخيراً، يتم فك تشفير التنبؤ الكامن باستخدام فك تشفير VAE إلى مساحة البكسل.

في هذه الحالة، يتم تخصيص مكالمة من أصل ثلاث وحدات نمطية لتحويلها من مساحة البكسل واللاتينية لـ LDM.

نظرًا لأن المساحة الكامنة لـ Marigold متوافقة مع Stable Diffusion الأساسي، فمن الممكن تسريع مكالمة خط الأنابيب بأكثر من 3x (85 مللي ثانية على RTX 3090) باستخدام [بديل خفيف الوزن لـ SD VAE](../api/models/autoencoder_tiny):

كما هو مقترح في [التحسينات](../optimization/torch2.0#torch.compile)، قد يؤدي إضافة 'torch.compile` إلى ضغط الأداء الإضافي اعتمادًا على الأجهزة المستهدفة:

### مقارنة نوعية مع العمق بأي

مع تحسينات السرعة أعلاه، توفر Marigold تنبؤات أكثر تفصيلاً وأسرع من [Depth Anything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything) مع نقطة تفتيش الأكبر [LiheYoung/depth-anything-large-hf](https://huggingface.co/LiheYoung/depth-anything-large-hf):

## زيادة الدقة وتجميعها

تحتوي خطوط أنابيب Marigold على آلية تجميع مدمجة تجمع بين عدة تنبؤات من الكمونات العشوائية المختلفة.

هذه طريقة وحشية لتحسين دقة التنبؤات، والاستفادة من الطبيعة التوليدية للانتشار.

يتم تنشيط مسار التجميع تلقائيًا عندما تكون قيمة وسيط 'ensemble_size' أكبر من '1'.

عند استهداف الدقة القصوى، من المنطقي ضبط 'num_inference_steps' في نفس الوقت مع 'ensemble_size'.

تختلف القيم الموصى بها عبر نقاط التفتيش ولكنها تعتمد بشكل أساسي على نوع الجدولة.

يظهر تأثير التجميع بشكل جيد مع المتجهات العادية للسطح:

كما هو موضح، فقد حصلت جميع المناطق ذات البنى الدقيقة، مثل الشعر، على تنبؤات أكثر تحفظًا وأكثر صحة في المتوسط.

هذه النتيجة أكثر ملاءمة للمهام الحساسة للدقة، مثل إعادة الإعمار ثلاثي الأبعاد.

### التقييم الكمي

لتقييم Marigold كميًا في لوحات القيادة والمعايير القياسية (مثل NYU وKITTI ومجموعات البيانات الأخرى)، اتبع بروتوكول التقييم الموصوف في الورقة: قم بتحميل نموذج الدقة الكاملة fp32 واستخدم القيم المناسبة لـ 'num_inference_steps' و 'ensemble_size'.

اختياريًا، قم بتهيئة العشوائية لضمان إمكانية إعادة الإنتاج. ستؤدي زيادة 'batch_size' إلى زيادة استخدام الجهاز إلى الحد الأقصى.

لقياس المقاييس، قم بتحميل الصورة التالية:
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين:

## استخدام عدم اليقين التنبئي
آلية التجميع المدمجة في خطوط أنابيب Marigold تجمع بين تنبؤات متعددة يتم الحصول عليها من latents عشوائية مختلفة.
وكأثر جانبي، يمكن استخدامه لقياس عدم اليقين النظري (النموذج)؛ ما عليك سوى تحديد ensemble_size أكبر من 1 وتعيين output_uncertainty=True.
سيتم توفر عدم اليقين الناتج في حقل "uncertainty" للإخراج.
يمكن تصويره على النحو التالي:

تفسير عدم اليقين سهل: تشير القيم الأعلى (اللون الأبيض) إلى البكسلات التي يناضل فيها النموذج من أجل تقديم تنبؤات متسقة.
من الواضح أن نموذج العمق أقل ثقة حول الحواف ذات الانقطاع، حيث يتغير عمق الكائن بشكل كبير.
ونموذج القواعد السطحية أقل ثقة في الهياكل الدقيقة، مثل الشعر، والمناطق الداكنة، مثل الياقة.

## معالجة الفيديو إطارًا تلو الآخر مع الاتساق الزمني
بسبب الطبيعة التوليدية لـ Marigold، يكون كل تنبؤ فريدًا ويتم تحديده بواسطة الضوضاء العشوائية التي تم أخذ عينات لها لتهيئة الكامن.
يصبح هذا عيبًا واضحًا مقارنة بشبكات الانحدار الكثيفة من النهاية إلى النهاية التقليدية، كما هو موضح في مقاطع الفيديو التالية:

لعنوانين هذا القسم، يمكنك الرجوع إلى الصور والفيديوهات في النص الأصلي.

لحل هذه المشكلة، يمكن تمرير حجة latents إلى خطوط الأنابيب، والتي تحدد نقطة البداية للانتشار.
وجدنا تجريبياً أن الجمع المحدب لنقطة البداية نفسها الضوضاء الكامنة والكامنة المقابلة لتنبؤ الإطار السابق يعطي نتائج سلسة بما فيه الكفاية، كما هو مطبق في المقتطف أدناه:

هنا، تبدأ عملية الانتشار من الكامن المحسوب.
يحدد خط الأنابيب output_latent=True للوصول إلى out.latent وحساب مساهمته في تهيئة الكامن للإطار التالي.
النتيجة الآن أكثر استقرارًا:

## Marigold لـ ControlNet
تطبيق شائع جدًا للتنبؤ بالعمق باستخدام نماذج الانتشار يأتي بالاقتران مع ControlNet.
تلعب وضوح العمق دورًا حاسمًا في الحصول على نتائج عالية الجودة من ControlNet.
كما هو موضح في المقارنات مع الطرق الأخرى أعلاه، تتفوق Marigold في تلك المهمة.
يوضح المقتطف أدناه كيفية تحميل صورة وحساب العمق وإدخالها في ControlNet بتنسيق متوافق:

النتيجة هي صورة تم إنشاؤها بواسطة ControlNet، مشروطة بالعمق والملفت: "صورة عالية الجودة لدراجة رياضية، مدينة".

آمل أن تجد Marigold مفيدة لحل مهامك النهائية، سواء كانت جزءًا من سير عمل توليدي أوسع أو مهمة إدراكية، مثل إعادة البناء ثلاثي الأبعاد.