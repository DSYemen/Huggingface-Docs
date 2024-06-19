## كيفية تشغيل Stable Diffusion مع Core ML 

Core ML هو تنسيق النموذج ومكتبة التعلم الآلي التي تدعمها Apple frameworks. إذا كنت مهتمًا بتشغيل نماذج Stable Diffusion داخل تطبيقات macOS أو iOS/iPadOS، فسيُظهر لك هذا الدليل كيفية تحويل نقاط تفتيش PyTorch الحالية إلى تنسيق Core ML واستخدامها للاستنتاج باستخدام Python أو Swift.

يمكن لنماذج Core ML الاستفادة من جميع محركات الحوسبة المتاحة في أجهزة Apple: وحدة المعالجة المركزية (CPU)، ووحدة معالجة الرسوميات (GPU)، وApple Neural Engine (أو ANE، وهو مسرع مُحسّن للنسيج متوفر في أجهزة Mac Silicon وأجهزة iPhone/iPad الحديثة). اعتمادًا على النموذج والجهاز الذي يعمل عليه، يمكن لـ Core ML أيضًا مزج ومواءمة محركات الحوسبة، لذا قد تعمل بعض أجزاء النموذج على وحدة المعالجة المركزية بينما تعمل أجزاء أخرى على وحدة معالجة الرسومات، على سبيل المثال.

## نقاط تفتيش Stable Diffusion Core ML

يتم تخزين أوزان Stable Diffusion (أو نقاط التفتيش) بتنسيق PyTorch، لذلك يجب تحويلها إلى تنسيق Core ML قبل أن نتمكن من استخدامها داخل التطبيقات الأصلية.

لحسن الحظ، قام مهندسو Apple بتطوير أداة تحويل بناءً على "diffusers" لتحويل نقاط تفتيش PyTorch إلى Core ML.

ومع ذلك، قبل تحويل النموذج، خذ لحظة لاستكشاف Hugging Face Hub - من المحتمل أن يكون النموذج الذي تهتم به متاحًا بالفعل بتنسيق Core ML:

- تشمل منظمة [Apple](https://huggingface.co/apple) إصدارات Stable Diffusion 1.4 و1.5 و2.0 base و2.1 base
- تشمل [مجتمع coreml](https://huggingface.co/coreml-community) نماذج مخصصة مُدربة
- استخدم هذا [الفلتر](https://huggingface.co/models?pipeline_tag=text-to-image&library=coreml&p=2&sort=likes) لإرجاع جميع نقاط تفتيش Core ML المتاحة

إذا لم تتمكن من العثور على النموذج الذي تبحث عنه، فنحن نوصي باتباع التعليمات الخاصة بـ [تحويل النماذج إلى Core ML](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) من Apple.

## تحديد متغير Core ML الذي سيتم استخدامه

يمكن تحويل نماذج Stable Diffusion إلى متغيرات Core ML مختلفة مخصصة لأغراض مختلفة:

- نوع كتل الاهتمام المستخدمة. يتم استخدام عملية الاهتمام "للاحتفاظ" بالعلاقة بين المناطق المختلفة في تمثيلات الصور وفهم كيفية ارتباط تمثيلات الصور والنصوص. الاهتمام كثيف الاستخدام للحوسبة والذاكرة، لذلك توجد تنفيذه المختلفة التي تأخذ في الاعتبار خصائص الأجهزة المختلفة. بالنسبة لنماذج Core ML Stable Diffusion، هناك متغيران للاهتمام:
* `split_einsum` ([قدمتها Apple](https://machinelearning.apple.com/research/neural-engine-transformers)) مُحسّنة لأجهزة ANE، والتي تتوفر في أجهزة iPhone وiPad وأجهزة الكمبيوتر من سلسلة M الحديثة.
* الاهتمام "الأصلي" (التنفيذ الأساسي المستخدم في "diffusers") متوافق فقط مع وحدة المعالجة المركزية/وحدة معالجة الرسومات وليس ANE. قد يكون *أسرع* لتشغيل نموذجك على وحدة المعالجة المركزية + وحدة معالجة الرسومات باستخدام الاهتمام "الأصلي" من ANE. راجع [هذا المعيار القياسي للأداء](https://huggingface.co/blog/fast-mac-diffusers#performance-benchmarks) بالإضافة إلى بعض [التدابير الإضافية التي قدمها المجتمع](https://github.com/huggingface/swift-coreml-diffusers/issues/31) للحصول على تفاصيل إضافية.

- إطار عمل الاستدلال المدعوم.
* "packages" مناسبة للاستدلال Python. يمكن استخدام هذا لاختبار نماذج Core ML المحولة قبل محاولة دمجها داخل التطبيقات الأصلية، أو إذا كنت ترغب في استكشاف أداء Core ML ولكنك لا تحتاج إلى دعم التطبيقات الأصلية. على سبيل المثال، يمكن لتطبيق بواجهة ويب استخدام backend Python Core ML بشكل مثالي.
* مطلوب نماذج "compiled" للرمز Swift. تنقسم نماذج "compiled" في Hub إلى أوزان نموذج UNet كبيرة في عدة ملفات للتوافق مع أجهزة iOS وiPadOS. وهذا يتوافق مع خيار التحويل [`--chunk-unet`](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml). إذا كنت تريد دعم التطبيقات الأصلية، فيجب عليك تحديد المتغير "compiled".

تتضمن النماذج الرسمية لـ Core ML Stable Diffusion [هذه المتغيرات](https://huggingface.co/apple/coreml-stable-diffusion-v1-4/tree/main)، ولكن قد تختلف نماذج المجتمع:

```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
├── compiled
└── packages
```

يمكنك تنزيل واستخدام المتغير الذي تحتاجه كما هو موضح أدناه.

## الاستدلال Core ML في Python

قم بتثبيت المكتبات التالية لتشغيل الاستدلال Core ML في Python:

```bash
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```

### تنزيل نقاط تفتيش النموذج

لتشغيل الاستدلال في Python، استخدم إحدى الإصدارات المخزنة في مجلدات "packages" لأن الإصدارات "compiled" متوافقة فقط مع Swift. يمكنك اختيار ما إذا كنت تريد استخدام الاهتمام "الأصلي" أو "split_einsum".

هكذا يمكنك تنزيل متغير الاهتمام "الأصلي" من Hub إلى دليل يسمى "models":

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### الاستدلال

بمجرد تنزيل لقطة من النموذج، يمكنك اختباره باستخدام نص Python الخاص بـ Apple.

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i models/coreml-stable-diffusion-v1-4_original_packages -o </path/to/output/image> --compute-unit CPU_AND_GPU --seed 93
```

مرر مسار نقطة التفتيش التي تم تنزيلها باستخدام علم `-i` إلى النص البرمجي. يشير `--compute-unit` إلى الأجهزة التي تريد السماح لها بالاستدلال. يجب أن يكون أحد الخيارات التالية: `ALL`، `CPU_AND_GPU`، `CPU_ONLY`، `CPU_AND_NE`. يمكنك أيضًا توفير مسار إخراج اختياري، وبذرة للتكرار.

يفترض نص الاستدلال أنك تستخدم الإصدار الأصلي من نموذج Stable Diffusion، `CompVis/stable-diffusion-v1-4`. إذا كنت تستخدم نموذجًا آخر، فيجب عليك تحديد معرف Hub الخاص به في سطر أوامر الاستدلال، باستخدام خيار `--model-version`. يعمل هذا لكل من النماذج المدعومة مسبقًا والنماذج المخصصة التي قمت بتدريبها أو ضبطها بدقة بنفسك.

على سبيل المثال، إذا كنت تريد استخدام [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5):

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version runwayml/stable-diffusion-v1-5
```

## الاستدلال Core ML في Swift

إن تشغيل الاستدلال في Swift أسرع قليلاً من Python لأن النماذج مجمعة بالفعل بتنسيق `mlmodelc`. يلاحظ هذا عند بدء تشغيل التطبيق عندما يتم تحميل النموذج ولكنه لا يلاحظ إذا قمت بتشغيل عدة أجيال بعد ذلك.

### التنزيل

لتشغيل الاستدلال في Swift على جهاز Mac الخاص بك، تحتاج إلى إحدى إصدارات نقطة التفتيش "compiled". نوصي بتنزيلها محليًا باستخدام رمز Python مشابه للمثال السابق، ولكن باستخدام أحد متغيرات "compiled":

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### الاستدلال

لتشغيل الاستدلال، يرجى استنساخ مستودع Apple:

```bash
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```

ثم استخدم أداة سطر الأوامر الخاصة بـ Apple، [Swift Package Manager](https://www.swift.org/package-manager/#):

```bash
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```

يجب عليك تحديد أحد نقاط التفتيش التي تم تنزيلها في الخطوة السابقة في `--resource-path`، لذا يرجى التأكد من احتوائها على حزم Core ML مجمعة بالملحق `.mlmodelc`. يجب أن تكون `--compute-units` إحدى هذه القيم: `all`، `cpuOnly`، `cpuAndGPU`، `cpuAndNeuralEngine`.

لمزيد من التفاصيل، يرجى الرجوع إلى [التعليمات في مستودع Apple](https://github.com/apple/ml-stable-diffusion).
## ميزات الناشرات المدعومة

لا تدعم نماذج Core ML ورمز الاستنتاج العديد من الميزات والخيارات والمرونة في الناشرات. فيما يلي بعض القيود التي يجب مراعاتها:

- تصلح نماذج Core ML للاستنتاج فقط. لا يمكن استخدامها للتدريب أو الضبط الدقيق.
- تم نقل جدولي مواعيد فقط إلى Swift، وهما الجدول الافتراضي الذي يستخدمه Stable Diffusion و "DPMSolverMultistepScheduler"، والذي قمنا بنقله إلى Swift من تنفيذنا "diffusers". نوصي باستخدام "DPMSolverMultistepScheduler"، حيث ينتج نفس الجودة في حوالي نصف الخطوات.
- تتوفر ميزات مثل النصوص السلبية، ومقياس التوجيه الخالي من التصنيف، والمهام من صورة إلى صورة في رمز الاستنتاج. لا تزال الميزات المتقدمة مثل التوجيه العميق، وControlNet، وlatent upscalers غير متوفرة.

يهدف مستودع Apple [conversion and inference repo](https://github.com/apple/ml-stable-diffusion) ومستودعنا الخاص [swift-coreml-diffusers](https://github.com/huggingface/swift-coreml-diffusers) إلى أن يكونا عارضين للتكنولوجيا لتمكين المطورين الآخرين من البناء عليها.

إذا كنت تشعر بشدة بشأن أي ميزات مفقودة، فيرجى فتح طلب ميزة أو، الأفضل من ذلك، إرسال طلب سحب المساهمة 🙂.

## تطبيق الناشرات الأصلي لـ Swift

تتمثل إحدى الطرق السهلة لتشغيل Stable Diffusion على أجهزة Apple الخاصة بك في استخدام مستودع Swift مفتوح المصدر [هنا](https://github.com/huggingface/swift-coreml-diffusers)، استنادًا إلى "diffusers" ومستودع Apple للتحويل والاستنتاج. يمكنك دراسة الكود، وتجميعه باستخدام Xcode وتكييفه وفقًا لاحتياجاتك. لراحتك، هناك أيضًا [تطبيق Mac مستقل في App Store](https://apps.apple.com/app/diffusers/id1666309574)، حتى تتمكن من اللعب به دون التعامل مع الكود أو IDE. إذا كنت مطورًا وقررت أن Core ML هو أفضل حل لبناء تطبيق Stable Diffusion الخاص بك، فيمكنك استخدام بقية هذا الدليل للبدء في مشروعك. لا يمكننا الانتظار لمعرفة ما ستبنيه 🙂.