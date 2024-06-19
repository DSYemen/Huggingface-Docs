## أساليب وأدوات للتدريب الفعال على وحدة معالجة الرسوميات (GPU) الفردية

يُظهر هذا الدليل التقنيات العملية التي يمكنك استخدامها لزيادة كفاءة تدريب نموذجك من خلال تحسين استخدام الذاكرة، أو تسريع التدريب، أو كليهما. إذا كنت ترغب في فهم كيفية استخدام وحدة معالجة الرسوميات أثناء التدريب، يُرجى الرجوع أولاً إلى الدليل المفاهيمي [تشريح تدريب النموذج](model_memory_anatomy). يركز هذا الدليل على التقنيات العملية.

عند تدريب النماذج الكبيرة، هناك جانبان يجب مراعاتهما في نفس الوقت:

- *معدل نقل البيانات/وقت التدريب*: يؤدي تحقيق أقصى معدل نقل (عينات/ثانية) إلى خفض تكلفة التدريب. ويتم تحقيق ذلك بشكل عام من خلال الاستفادة من وحدة معالجة الرسوميات قدر الإمكان، وبالتالي ملء ذاكرة وحدة معالجة الرسوميات إلى حدّها الأقصى. إذا تجاوز حجم الدُفعة المطلوب حدود ذاكرة وحدة معالجة الرسوميات، فيمكن لتقنيات تحسين الذاكرة، مثل تراكم التدرجات، أن تساعد في ذلك.

- *أداء النموذج*: ومع ذلك، إذا كان حجم الدفعة المفضل يناسب الذاكرة، فلا يوجد سبب لتطبيق تقنيات تحسين الذاكرة لأنها يمكن أن تبطئ التدريب. لمجرد أن المرء يمكنه استخدام حجم دفعة كبير، لا يعني بالضرورة أنه ينبغي عليه ذلك. كجزء من ضبط المعلمات، يجب عليك تحديد حجم الدفعة الذي يحقق أفضل النتائج، ثم تحسين الموارد وفقًا لذلك.

يمكن تصنيف الأساليب والأدوات المشمولة في هذا الدليل بناءً على التأثير الذي تحدثه على عملية التدريب:

| الأسلوب/الأداة | تحسين سرعة التدريب | تحسين استخدام الذاكرة |
| :--------------------------- | :------------------------ | :----------------------------- |
| [اختيار حجم الدفعة](#batch-size-choice) | نعم | نعم |
| [تراكم التدرجات](#gradient-accumulation) | لا | نعم |
| [التدقيق التدريجي](#gradient-checkpointing) | لا | نعم |
| [تدريب الدقة المختلطة](#mixed-precision-training) | نعم | (لا) |
| [اختيار المحسن](#optimizer-choice) | نعم | نعم |
| [التحميل المسبق للبيانات](#data-preloading) | نعم | لا |
| [DeepSpeed Zero](#deepspeed-zero) | لا | نعم |
| [torch.compile](#using-torchcompile) | نعم | لا |
| [ضبط دقيق فعال للمعلمات (PEFT)](#using--peft) | لا | نعم |

يمكنك الجمع بين الأساليب المذكورة أعلاه للحصول على تأثير تراكمي. تتوفر هذه التقنيات سواء كنت تدرب نموذجك باستخدام [`Trainer`] أو تكتب حلقة PyTorch نقية، وفي هذه الحالة يمكنك [تكوين هذه التحسينات باستخدام 🤗 Accelerate](#using--accelerate).

إذا لم تؤد هذه الأساليب إلى مكاسب كافية، فيمكنك استكشاف الخيارات التالية:

- [الاطلاع على بناء حاوية Docker مخصصة خاصة بك مع برامج البناء الفعالة مسبقًا](#efficient-software-prebuilds)
- [النظر في نموذج يستخدم مزيجًا من الخبراء (MoE)](#mixture-of-experts)
- [تحويل نموذجك إلى BetterTransformer للاستفادة من الاهتمام الأصلي في PyTorch](#using-pytorch-native-attention-and-flash-attention)

أخيرًا، إذا لم يكن كل ما سبق كافيًا، حتى بعد التبديل إلى وحدة معالجة رسوميات من فئة الخوادم مثل A100، ففكر في الانتقال إلى إعداد متعدد وحدات معالجة الرسوميات. لا تزال جميع هذه الأساليب صالحة في إعداد متعدد وحدات معالجة الرسوميات، بالإضافة إلى ذلك، يمكنك الاستفادة من تقنيات التوازي الإضافية الموضحة في قسم [متعدد وحدات معالجة الرسوميات](perf_train_gpu_many).

## اختيار حجم الدفعة

لتحقيق الأداء الأمثل، ابدأ بتحديد حجم الدفعة المناسب. يُنصح باستخدام أحجام دفعات وأعداد العصبونات للإدخال/الإخراج التي تكون من حجم 2^N. غالبًا ما يكون هذا العدد مضاعفًا لـ 8، ولكنه يمكن أن يكون أعلى اعتمادًا على الأجهزة المستخدمة ونوع بيانات النموذج.

للاطلاع على ذلك، راجع توصية NVIDIA الخاصة بـ [أعداد العصبونات للإدخال/الإخراج](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features) و[حجم الدفعة](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size) للطبقات المتصلة بالكامل (التي تشارك في عمليات الضرب المصفوفي العام).

تحدد [متطلبات Tensor Core](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) المضاعف بناءً على نوع بيانات الأجهزة والأجهزة. على سبيل المثال، بالنسبة لنوع بيانات fp16، يوصى باستخدام مضاعف 8، ما لم تكن وحدة معالجة الرسوميات من نوع A100، وفي هذه الحالة استخدم مضاعفات 64.

بالنسبة للمعلمات الصغيرة، ضع في اعتبارك أيضًا [تأثيرات التكميم البعدي](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization). هنا يحدث التبليط ويمكن أن يؤدي المضاعف الصحيح إلى تسريع كبير.

## تراكم التدرجات

يهدف أسلوب **تراكم التدرجات** إلى حساب التدرجات على دفعات أصغر بدلاً من حسابها للدفعة بأكملها مرة واحدة. ينطوي هذا النهج على حساب التدرجات بشكل تكراري على دفعات أصغر من خلال تنفيذ عمليات تمرير للأمام والخلف عبر النموذج وتراكم التدرجات أثناء هذه العملية. بمجرد تراكم عدد كافٍ من التدرجات، يتم تنفيذ خطوة التحسين الخاصة بالنموذج. من خلال استخدام تراكم التدرجات، يصبح من الممكن زيادة **حجم الدفعة الفعال** إلى ما بعد القيود التي تفرضها سعة ذاكرة وحدة معالجة الرسوميات. ومع ذلك، من المهم ملاحظة أن عمليات التمرير الإضافية للأمام والخلف التي قدمها تراكم التدرجات يمكن أن تبطئ عملية التدريب.

يمكنك تمكين تراكم التدرجات عن طريق إضافة حجة `gradient_accumulation_steps` إلى [`TrainingArguments`]:

```بايثون
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
```

في المثال أعلاه، يصبح حجم الدفعة الفعال لديك 4.

أو، استخدم 🤗 Accelerate للتحكم الكامل في حلقة التدريب. يمكنك العثور على مثال 🤗 Accelerate [أدناه في هذا الدليل](#using--accelerate).

في حين يُنصح بزيادة استخدام وحدة معالجة الرسوميات إلى الحد الأقصى قدر الإمكان، يمكن أن يؤدي العدد الكبير من خطوات تراكم التدرجات إلى تباطؤ تدريب أكثر وضوحًا. ضع في اعتبارك المثال التالي. لنفترض أن `per_device_train_batch_size=4` بدون تراكم التدرجات يصل إلى حد وحدة معالجة الرسوميات. إذا كنت ترغب في التدريب باستخدام دفعات من حجم 64، فلا تقم بتعيين `per_device_train_batch_size` إلى 1 و`gradient_accumulation_steps` إلى 64. بدلاً من ذلك، احتفظ بـ `per_device_train_batch_size=4` وقم بتعيين `gradient_accumulation_steps=16`. يؤدي هذا إلى نفس حجم الدفعة الفعال مع الاستفادة بشكل أفضل من موارد وحدة معالجة الرسوميات المتاحة.

للحصول على معلومات إضافية، يُرجى الرجوع إلى معايير حجم الدفعة وتراكم التدرجات لـ [RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537) و[A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957).

## التدقيق التدريجي للتدرجات

قد تواجه بعض النماذج الكبيرة مشكلات في الذاكرة حتى عند تعيين حجم الدفعة إلى 1 واستخدام تراكم التدرجات. ويرجع ذلك إلى وجود مكونات أخرى تتطلب أيضًا مساحة تخزين في الذاكرة.

يمكن أن يؤدي حفظ جميع التنشيطات من عملية التمرير للأمام من أجل حساب التدرجات أثناء التمرير للخلف إلى زيادة كبيرة في الذاكرة. ومن شأن النهج البديل المتمثل في التخلص من التنشيطات وإعادة حسابها عند الحاجة أثناء التمرير للخلف أن يُدخل زيادة كبيرة في الحوسبة تبطئ عملية التدريب.

يقدم **التدقيق التدريجي للتدرجات** حل وسط بين هذين النهجين، ويحفظ تنشيطات مختارة بشكل استراتيجي في جميع أنحاء الرسم البياني الحسابي، بحيث لا يلزم إعادة حساب سوى جزء من التنشيطات للتدرجات. للحصول على شرح متعمق للتدقيق التدريجي للتدرجات، راجع [هذه المقالة الرائعة](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9).

لتمكين التدقيق التدريجي للتدرجات في [`Trainer`]، قم بتمرير العلم المقابل إلى [`TrainingArguments`]:

```بايثون
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
```

أو، استخدم 🤗 Accelerate - يمكنك العثور على مثال 🤗 Accelerate [أدناه في هذا الدليل](#using--accelerate).

<نصيحة>

في حين أن التدقيق التدريجي للتدرجات قد يحسن كفاءة الذاكرة، إلا أنه يبطئ التدريب بنسبة 20% تقريبًا.

</نصيحة>
## التدريب ذو الدقة المختلطة

تعد **التدريب ذو الدقة المختلطة**  تقنية تهدف إلى تحسين الكفاءة الحسابية لتدريب النماذج من خلال استخدام تنسيقات رقمية أقل دقة لبعض المتغيرات. تقليديًا، تستخدم معظم النماذج دقة النقطة العائمة 32 بت (fp32 أو float32) لتمثيل ومعالجة المتغيرات. ومع ذلك، لا تتطلب جميع المتغيرات هذا المستوى العالي من الدقة لتحقيق نتائج دقيقة. من خلال تقليل دقة بعض المتغيرات إلى تنسيقات رقمية أقل، مثل النقطة العائمة 16 بت (fp16 أو float16)، يمكننا تسريع العمليات الحسابية. نظرًا لأن بعض العمليات الحسابية في هذا النهج يتم إجراؤها بنصف الدقة، بينما لا تزال بعضها الآخر بدقة كاملة، يُطلق على النهج اسم التدريب ذو الدقة المختلطة.

يتم تحقيق التدريب ذو الدقة المختلطة في معظم الأحيان باستخدام أنواع بيانات fp16 (float16)، ومع ذلك، توفر بعض بنيات GPU (مثل بنية Ampere) أنواع بيانات bf16 وtf32 (نوع بيانات CUDA الداخلي). تحقق من [مدونة NVIDIA](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) لمعرفة المزيد عن الاختلافات بين أنواع البيانات هذه.

### fp16

تأتي الميزة الرئيسية للتدريب ذو الدقة المختلطة من حفظ التنشيطات بنصف الدقة (fp16). على الرغم من أن التدرجات يتم حسابها أيضًا بنصف الدقة، إلا أنها تُحول مرة أخرى إلى دقة كاملة لخطوة التحسين بحيث لا يتم توفير الذاكرة هنا.

في حين أن التدريب ذو الدقة المختلطة يؤدي إلى عمليات حسابية أسرع، إلا أنه يمكن أن يؤدي أيضًا إلى استخدام المزيد من ذاكرة GPU، خاصة لحجم الدفعات الصغيرة. ويرجع ذلك إلى أن النموذج موجود الآن على GPU بدقة 16 بت و32 بت (1.5x من النموذج الأصلي على GPU).

لتمكين التدريب ذو الدقة المختلطة، قم بتعيين علامة `fp16` على `True`:

```بي
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)
```

إذا كنت تفضل استخدام 🤗 Accelerate، فابحث عن مثال 🤗 Accelerate [فيما بعد في هذا الدليل](#using-accelerate).

### BF16

إذا كان لديك إمكانية الوصول إلى Ampere أو أجهزة أحدث، فيمكنك استخدام bf16 للتدريب والتقييم ذي الدقة المختلطة. في حين أن دقة bf16 أسوأ من fp16، إلا أن لها نطاق ديناميكي أكبر بكثير. في fp16 أكبر رقم يمكن أن يكون هو `65535` وأي رقم أعلى من ذلك سيؤدي إلى فيض. يمكن أن يكون رقم bf16 كبيرًا مثل `3.39e+38` (!) وهو ما يماثل تقريبًا fp32 - لأن كليهما يستخدم 8 بتات للنطاق الرقمي.

يمكنك تمكين BF16 في 🤗 Trainer باستخدام ما يلي:

```بايثون
training_args = TrainingArguments(bf16=True, **default_args)
```

### TF32

تستخدم أجهزة Ampere نوع بيانات سحري يسمى tf32. لديها نفس النطاق الرقمي مثل fp32 (8 بت)، ولكن بدلاً من 23 بت من الدقة، لديها فقط 10 بتات (نفس fp16) وتستخدم فقط 19 بت في المجموع. إنه "سحري" بمعنى أنه يمكنك استخدام رمز التدريب و/أو الاستدلال fp32 العادي، ومن خلال تمكين دعم tf32، يمكنك الحصول على تحسن في الإنتاجية يصل إلى 3 مرات. كل ما عليك فعله هو إضافة ما يلي إلى رمزك:

```بايثون
استيراد الشعلة

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

سيقوم CUDA تلقائيًا بالتبديل إلى استخدام tf32 بدلاً من fp32 حيثما كان ذلك ممكنًا، بافتراض أن GPU المستخدمة هي من سلسلة Ampere.

وفقًا لبحث [NVIDIA](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)، فإن غالبية أعباء عمل التدريب على التعلم الآلي تظهر نفس الغموض والتقارب مع التدريب tf32 كما هو الحال مع fp32. إذا كنت تستخدم بالفعل fp16 أو bf16 الدقة المختلطة، فقد يساعد ذلك في الإنتاجية أيضًا.

يمكنك تمكين هذا الوضع في 🤗 Trainer:

```بايثون
TrainingArguments(tf32=True, **default_args)
```

<Tip>

لا يمكن الوصول إلى tf32 مباشرة عبر `tensor.to(dtype=torch.tf32)` لأنه نوع بيانات CUDA داخلي. تحتاج إلى `torch>=1.7` لاستخدام أنواع بيانات tf32.

</Tip>

للحصول على معلومات إضافية حول tf32 مقابل الدقة الأخرى، يرجى الرجوع إلى المعايير المرجعية التالية:
[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803) و [A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004543189).

## فلاش الانتباه 2

يمكنك تسريع الإنتاجية التدريبية من خلال استخدام تكامل Flash Attention 2 في المحولات. تحقق من القسم المناسب في [قسم GPU الفردي](./perf_infer_gpu_one#Flash-Attention-2) لمعرفة المزيد حول كيفية تحميل نموذج باستخدام وحدات Flash Attention 2.

## اختيار المحسن

المحسن الأكثر شيوعًا المستخدم لتدريب نماذج المحول هو Adam أو AdamW (Adam مع انخفاض الوزن). يحقق Adam تقاربًا جيدًا من خلال تخزين المتوسط المتحرك للتدرجات السابقة؛ ومع ذلك، فإنه يضيف بصمة ذاكرة إضافية بحجم عدد معلمات النموذج. لمعالجة ذلك، يمكنك استخدام محسن بديل.

على سبيل المثال، إذا كان لديك [NVIDIA/apex](https://github.com/NVIDIA/apex) مثبتًا لـ GPUs NVIDIA، أو [ROCmSoftwarePlatform/apex](https://github.com/ROCmSoftwarePlatform/apex) لـ GPUs AMD، فسيمنحك `adamw_apex_fused` أسرع تجربة تدريب بين جميع محسنات AdamW المدعومة.

يضم [`Trainer`] مجموعة متنوعة من المحسنات التي يمكن استخدامها مباشرة من الصندوق: `adamw_hf`، `adamw_torch`، `adamw_torch_fused`، `adamw_apex_fused`، `adamw_anyprecision`، `adafactor`، أو `adamw_bnb_8bit`. يمكن توصيل المزيد من المحسنات عبر تنفيذ جهة خارجية.

دعنا نلقي نظرة فاحصة على بديلين لمحسن AdamW:

1. `adafactor` المتاح في [`Trainer`]
2. `adamw_bnb_8bit` متاح أيضًا في Trainer، ولكن يتم توفير التكامل مع جهة خارجية أدناه للتوضيح.

لمقارنة ذلك، بالنسبة لنموذج معلمات 3B، مثل "google-t5/t5-3b":

* سوف يحتاج محسن AdamW القياسي إلى 24 جيجابايت من ذاكرة GPU لأنه يستخدم 8 بايتات لكل معلمة (8*3 => 24 جيجابايت)
* سوف يحتاج محسن Adafactor إلى أكثر من 12 جيجابايت. يستخدم أكثر بقليل من 4 بايتات لكل معلمة، لذا 4*3 ثم بعض الإضافات.
* سوف يستخدم محسن 8 بت BNB كمية صغيرة فقط (2*3) 6 جيجابايت إذا تم تحويل جميع حالات المحسن.

### Adafactor

لا يقوم Adafactor بتخزين المتوسطات المتحركة لكل عنصر في المصفوفات الوزنية. بدلاً من ذلك، فإنه يحتفظ بالمعلومات المجمعة (مجاميع المتوسطات المتحركة صفًا وعموديًا)، مما يقلل بشكل كبير من بصمته. ومع ذلك، مقارنة بـ Adam، قد يكون لدى Adafactor تقارب أبطأ في بعض الحالات.

يمكنك التبديل إلى Adafactor عن طريق تعيين `optim="adafactor"` في [`TrainingArguments`]:

```بي
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor"، **default_args)
```

بالجمع بين النهج الأخرى (تراكم التدرجات، ومراجعة التدرجات، والتدريب ذو الدقة المختلطة)، يمكنك ملاحظة تحسن يصل إلى 3 مرات مع الحفاظ على الإنتاجية! ومع ذلك، كما ذكرنا سابقًا، قد يكون تقارب Adafactor أسوأ من Adam.

### Adam 8 بت

بدلاً من تجميع حالات المحسن مثل Adafactor، يحتفظ Adam 8 بت بالحالة الكاملة ويقوم بتحويلها إلى النطاق. يعني التحويل أنه يتم تخزين الحالة بدقة أقل ويتم إلغاء تحويلها فقط للتحسين. هذا مشابه للفكرة وراء التدريب ذو الدقة المختلطة.

لاستخدام `adamw_bnb_8bit`، ما عليك سوى تعيين `optim="adamw_bnb_8bit"` في [`TrainingArguments`]:

```بي
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adamw_bnb_8bit"، **default_args)
```

ومع ذلك، يمكننا أيضًا استخدام تنفيذ جهة خارجية لمحسن 8 بت للتوضيح على كيفية دمج ذلك.

أولاً، اتبع دليل التثبيت في مستودع GitHub [repo](https://github.com/TimDettmers/bitsandbytes) لتثبيت مكتبة `bitsandbytes` التي تنفذ محسن Adam 8 بت.

بعد ذلك، تحتاج إلى تهيئة المحسن. ينطوي ذلك على خطوتين:

* أولاً، قم بتقسيم معلمات النموذج إلى مجموعتين - واحدة يتم تطبيق انخفاض الوزن عليها، والأخرى لا. عادةً ما يتم استبعاد الانحياز ومعلمات طبقة التطبيع من انخفاض الوزن.
* ثم قم بتنظيف الحجج لاستخدام نفس المعلمات كما هو الحال في محسن AdamW المستخدم سابقًا.

```بي
استيراد بتساندبايتس باسم بتب

من الشعلة استيراد نن

من transformers.trainer_pt_utils استيراد get_parameter_names

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
{
"params": [p for n, p in model.named_parameters() if n in decay_parameters],
"weight_decay": training_args.weight_decay,
},
{
"params": [p for n, p in model.named_parameters() if n not in decay_parameters],
"weight_decay": 0.0,
},
]

optimizer_kwargs = {
"betas": (training_args.adam_beta1, training_args.adam_beta2)،
"eps": training_args.adam_epsilon
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
optimizer_grouped_parameters,
betas=(training_args.adam_beta1, training_args.adam_beta2)،
eps=training_args.adam_epsilon,
lr=training_args.learning_rate,
)
```

أخيرًا، قم بتمرير المحسن المخصص كحجة إلى `Trainer`:

```بي
المدرب = Trainer(model=model، args=training_args، train_dataset=ds، optimizers=(adam_bnb_optim، None))
```

بالجمع بين النهج الأخرى (تراكم التدرجات، ومراجعة التدرجات، والتدريب ذو الدقة المختلطة)، يمكنك توقع الحصول على تحسن في الذاكرة يبلغ حوالي 3 مرات وحتى زيادة طفيفة في الإنتاجية كما هو الحال عند استخدام Adafactor.

### multi_tensor

قدم pytorch-nightly `torch.optim._multi_tensor` الذي يجب أن يسرع بشكل كبير المحسنات للمواقف التي تحتوي على العديد من الميزات الصغيرة. يجب أن يصبح هذا هو الافتراضي في النهاية، ولكن إذا كنت تريد تجربته في وقت أقرب، فراجع مشكلة GitHub هذه [issue](https://github.com/huggingface/transformers/issues/9965).
## التحميل المسبق للبيانات

من المتطلبات المهمة للوصول إلى سرعة تدريب عالية هي القدرة على تغذية وحدة معالجة الرسوميات (GPU) بأقصى سرعة يمكنها التعامل معها. بشكل افتراضي، يحدث كل شيء في العملية الرئيسية، وقد لا تتمكن من قراءة البيانات من القرص بسرعة كافية، مما يؤدي إلى اختناق يؤدي إلى الاستخدام الناقص لوحدة معالجة الرسوميات. قم بتكوين الحجج التالية للحد من الاختناق:

- `DataLoader(pin_memory=True, ...)` - يضمن تحميل البيانات مسبقًا في الذاكرة المثبتة على وحدة المعالجة المركزية (CPU) ويؤدي عادةً إلى نقل أسرع بكثير من ذاكرة وحدة المعالجة المركزية إلى ذاكرة وحدة معالجة الرسوميات.

- `DataLoader(num_workers=4, ...)` - قم بتشغيل عدة عمال لتحميل البيانات بشكل أسرع. أثناء التدريب، راقب إحصائيات استخدام وحدة معالجة الرسوميات؛ إذا كان بعيدًا عن 100%، فجرّب زيادة عدد العمال. بالطبع، قد تكون المشكلة في مكان آخر، لذلك قد لا يؤدي زيادة عدد العمال إلى تحسين الأداء.

عند استخدام [`Trainer`]، تكون حجج [`TrainingArguments`] المقابلة هي: `dataloader_pin_memory` (`True` بشكل افتراضي)، و`dataloader_num_workers` (افتراضيًا `0`).

## DeepSpeed ZeRO

DeepSpeed هي مكتبة تحسين للتعلم العميق مفتوحة المصدر مدمجة مع 🤗 Transformers و🤗 Accelerate. يوفر مجموعة واسعة من الميزات والتحسينات المصممة لتحسين كفاءة ونطاق تدريب التعلم العميق واسع النطاق.

إذا كان نموذجك يناسب وحدة معالجة الرسوميات (GPU) واحدة ولديك مساحة كافية لتناسب حجم دفعة صغير، فلا تحتاج إلى استخدام DeepSpeed لأنه سيبطئ الأمور فقط. ومع ذلك، إذا لم يناسب النموذج وحدة معالجة الرسوميات (GPU) واحدة أو لا يمكنك تناسب حجم دفعة صغير، فيمكنك الاستفادة من DeepSpeed ZeRO + CPU Offload، أو NVMe Offload للنماذج الأكبر حجمًا. في هذه الحالة، تحتاج إلى تثبيت المكتبة بشكل منفصل، ثم اتبع أحد الأدلة لإنشاء ملف تكوين وتشغيل DeepSpeed:

* للحصول على دليل متعمق حول تكامل DeepSpeed مع [`Trainer`]`، راجع [وثائق](main_classes/deepspeed) المقابلة، وتحديدًا [القسم الخاص بوحدة معالجة الرسوميات (GPU) واحدة](main_classes/deepspeed#deployment-with-one-gpu). مطلوب بعض التعديلات لاستخدام DeepSpeed في دفتر ملاحظات؛ يرجى إلقاء نظرة على [الدليل المقابل](main_classes/deepspeed#deployment-in-notebooks).

* إذا كنت تفضل استخدام 🤗 Accelerate، يرجى الرجوع إلى [دليل DeepSpeed في 🤗 Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed).

## استخدام torch.compile

قدم PyTorch 2.0 دالة تجميع جديدة لا تتطلب أي تعديل على كود PyTorch الموجود ولكن يمكنها تحسين كودك عن طريق إضافة سطر واحد من الكود: `model = torch.compile(model)`.

إذا كنت تستخدم [`Trainer`]`، فأنت بحاجة فقط إلى تمرير` `torch_compile` option في [`TrainingArguments`]:

```python
training_args = TrainingArguments(torch_compile=True, **default_args)
```

يستخدم `torch.compile` واجهة برمجة تطبيقات تقييم الإطار الخاصة بـ Python لإنشاء رسم بياني تلقائيًا من برامج PyTorch الموجودة. بعد التقاط الرسم البياني، يمكن نشر backends مختلفة لخفض الرسم البياني إلى محرك محسّن.

يمكنك العثور على مزيد من التفاصيل والاختبارات المعيارية في [وثائق PyTorch](https://pytorch.org/get-started/pytorch-2.0/).

لدى `torch.compile` قائمة متزايدة من backends، والتي يمكن العثور عليها عن طريق استدعاء `torchdynamo.list_backends()`، لكل منها تبعياته الاختيارية.

حدد backend الذي تريد استخدامه عن طريق تحديده عبر `torch_compile_backend` في [`TrainingArguments`]. بعض من backends الأكثر استخدامًا هي:

**backends التصحيح**:

* `dynamo.optimize("eager")` - يستخدم PyTorch لتشغيل GraphModule المستخرج. هذا مفيد جدًا في تصحيح مشكلات TorchDynamo.

* `dynamo.optimize("aot_eager")` - يستخدم AotAutograd بدون مترجم، أي باستخدام Eager الخاص بـ PyTorch للرسوم البيانية للأمام والخلف المستخرجة من AotAutograd. هذا مفيد للتصحيح، ومن غير المرجح أن يعطي سرعات.

**backends التدريب والاستدلال**:

* `dynamo.optimize("inductor")` - يستخدم backend TorchInductor مع AotAutograd وcudagraphs عن طريق الاستفادة من نواة Triton المرمزة [اقرأ المزيد](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)

* `dynamo.optimize("nvfuser")` - nvFuser مع TorchScript. [اقرأ المزيد](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)

* `dynamo.optimize("aot_nvfuser")` - nvFuser مع AotAutograd. [اقرأ المزيد](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)

* `dynamo.optimize("aot_cudagraphs")` - cudagraphs مع AotAutograd. [اقرأ المزيد](https://github.com/pytorch/torchdynamo/pull/757)

**backends الاستدلال فقط**:

* `dynamo.optimize("ofi")` - يستخدم Torchscript optimize_for_inference. [اقرأ المزيد](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)

* `dynamo.optimize("fx2trt")` - يستخدم NVIDIA TensorRT لتحسين الاستدلال. [اقرأ المزيد](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html)

* `dynamo.optimize("onnxrt")` - يستخدم ONNXRT للاستدلال على وحدة المعالجة المركزية/وحدة معالجة الرسوميات. [اقرأ المزيد](https://onnxruntime.ai/)

* `dynamo.optimize("ipex")` - يستخدم IPEX للاستدلال على وحدة المعالجة المركزية. [اقرأ المزيد](https://github.com/intel/intel-extension-for-pytorch)

لمثال على استخدام `torch.compile` مع 🤗 Transformers، تحقق من [منشور المدونة حول ضبط دقيق لنموذج BERT لتصنيف النصوص باستخدام أحدث ميزات PyTorch 2.0](https://www.philschmid.de/getting-started-pytorch-2.0-transformers)

## استخدام 🤗 PEFT

تقوم [طرق الضبط الدقيق الفعالة للمعلمات (PEFT)](https://huggingface.co/blog/peft) بتجميد معلمات النموذج المسبق التدريب أثناء الضبط الدقيق وإضافة عدد صغير من المعلمات القابلة للتدريب (المهايئات) فوقه.

ونتيجة لذلك، يتم تقليل [الذاكرة المرتبطة بحالات المُحَسِّن والتدرجات](https://huggingface.co/docs/transformers/model_memory_anatomy#anatomy-of-models-memory) بشكل كبير.

على سبيل المثال، مع AdamW الفانيلا، سيكون متطلب الذاكرة لحالة المُحَسِّن على النحو التالي:

* نسخة fp32 من المعلمات: 4 bytes/param

* الزخم: 4 bytes/param

* التباين: 4 bytes/param

لنفترض وجود نموذج به 7 مليارات معلمة و200 مليون معلمة تم حقنها باستخدام [المهايئات ذات الرتبة المنخفضة](https://huggingface.co/docs/peft/conceptual_guides/lora).

سيكون متطلب الذاكرة لحالة المُحَسِّن للنموذج العادي 12 * 7 = 84 جيجابايت (بافتراض 7 مليارات معلمة قابلة للتدريب).

تضيف Lora زيادة طفيفة في الذاكرة المرتبطة بأوزان النموذج وتقلل بشكل كبير من متطلبات الذاكرة لحالة المُحَسِّن إلى 12 * 0.2 = 2.4 جيجابايت.

اقرأ المزيد حول PEFT واستخدامه المفصل في [وثائق PEFT](https://huggingface.co/docs/peft/) أو [مستودع PEFT](https://github.com/huggingface/peft).

## استخدام 🤗 Accelerate

مع [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) يمكنك استخدام الطرق المذكورة أعلاه مع اكتساب التحكم الكامل في حلقة التدريب ويمكنك في الأساس كتابة الحلقة في PyTorch النقي مع بعض التعديلات الطفيفة.

لنفترض أنك قمت بدمج الطرق في [`TrainingArguments`] كما يلي:

```py
training_args = TrainingArguments(
per_device_train_batch_size=1,
gradient_accumulation_steps=4,
gradient_checkpointing=True,
fp16=True,
**default_args,
)
```

مثال حلقة التدريب الكامل مع 🤗 Accelerate عبارة عن عدد قليل من أسطر الكود:

```py
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

if training_args.gradient_checkpointing:
model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

model.train()
for step, batch in enumerate(dataloader, start=1):
loss = model(**batch).loss
loss = loss / training_args.gradient_accumulation_steps
accelerator.backward(loss)
if step % training_args.gradient_accumulation_steps == 0:
optimizer.step()
optimizer.zero_grad()
```

أولاً، نقوم بتغليف مجموعة البيانات في [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

بعد ذلك، يمكننا تمكين التحقق من التدرج عن طريق استدعاء طريقة [`~PreTrainedModel.gradient_checkpointing_enable`] للنموذج.

عند تهيئة [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator)

يمكننا تحديد ما إذا كنا نريد استخدام التدريب بالدقة المختلطة وسوف يعتني بذلك بالنسبة لنا في مكالمة [`prepare`]

أثناء مكالمة [`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare)

سيتم أيضًا توزيع dataloader عبر العمال إذا كنا نستخدم وحدات معالجة الرسوميات (GPU) متعددة. نستخدم [محسن 8 بت](#8-bit-adam) نفسه من المثال السابق.

أخيرًا، يمكننا إضافة حلقة التدريب الرئيسية. لاحظ أن مكالمة `backward` تتم من قبل 🤗 Accelerate. يمكننا أيضًا أن نرى كيف يعمل تراكم التدرج: نقوم بتطبيع الخسارة، لذا نحصل على المتوسط في نهاية التراكم وبمجرد أن يكون لدينا عدد كافٍ من الخطوات، نقوم بتشغيل التحسين.

لا يستغرق تنفيذ تقنيات التحسين هذه باستخدام 🤗 Accelerate سوى عدد قليل من أسطر الكود ويأتي بميزة المرونة في حلقة التدريب. للاطلاع على الوثائق الكاملة لجميع الميزات، راجع [وثائق Accelerate](https://huggingface.co/docs/accelerate/index).
## حزم البرامج المُسبقة الفعالة

يمكن أن يتطلب الأمر في بعض الأحيان بذل جهود إضافية لبناء بعض المكونات مسبقًا. على سبيل المثال، إذا كنت تستخدم مكتبات مثل "أبيكس" التي لا تأتي مجمعة مسبقًا. وفي مواقف أخرى، يمكن أن يكون من المعقد معرفة كيفية تثبيت حزمة "كودا تولكيت" المناسبة على مستوى النظام.

ولمعالجة هذه السيناريوهات، أصدرت "باي تورش" و"إنفيديا" نسخة جديدة من حاوية "إن جي سي" الخاصة بـ"داوكر" والتي تأتي بالفعل مع كل شيء مجمع مسبقًا. كل ما عليك هو تثبيت برامجك عليها، وستعمل على الفور.

وهذا النهج مفيد أيضًا إذا كنت ترغب في ضبط مصدر "باي تورش" و/أو إنشاء بناء مخصص جديد.

للعثور على إصدار صورة "داوكر" الذي تريده، ابدأ بـ[ملاحظات إصدار "باي تورش"](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)، واختر أحدث الإصدارات الشهرية. انتقل إلى ملاحظات الإصدار للإصدار المطلوب، وتحقق من أن مكونات البيئة تتطابق مع احتياجاتك (بما في ذلك متطلبات برنامج تشغيل "إنفيديا"!)، ثم في أعلى تلك الوثيقة، انتقل إلى صفحة "إن جي سي" المقابلة. إذا ضللت الطريق لأي سبب، فهذا هو [فهرس جميع صور باي تورش إن جي سي](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch).

بعد ذلك، اتبع التعليمات لتنزيل ونشر صورة "داوكر".

## مزيج الخبراء

أفادت بعض الأوراق البحثية الحديثة عن تسريع التدريب بمعدل 4-5 مرات وتسريع الاستدلال عن طريق دمج مزيج من الخبراء (MoE) في نماذج المحول.

ونظرًا لأنه قد اكتُشف أن المزيد من المعلمات تؤدي إلى أداء أفضل، تتيح هذه التقنية زيادة عدد المعلمات بمقدار عشرة أضعاف دون زيادة تكاليف التدريب.

وفي هذا النهج، يتم استبدال كل طبقة شبكة عصبية اصطناعية أخرى بطبقة MoE تتألف من العديد من الخبراء، مع دالة بوابة تقوم بتدريب كل خبير بطريقة متوازنة اعتمادًا على موضع رمز الإدخال في تسلسل.

يمكن العثور على تفاصيل ومقارنات شاملة في الأوراق البحثية المدرجة في نهاية هذا القسم.

الجانب السلبي الرئيسي لهذا النهج هو أنه يتطلب كميات مذهلة من ذاكرة وحدة معالجة الرسومات - أكبر بحوالي عشرة أضعاف من نظيرتها الكثيفة. وقد اقترحت عمليات تقطير ونهج مختلفة للتغلب على متطلبات الذاكرة الأعلى بكثير.

ولكن هناك مفاضلة مباشرة، حيث يمكنك استخدام عدد قليل فقط من الخبراء مع نموذج أساسي أصغر بمقدار 2-3 مرات بدلاً من عشرات أو مئات الخبراء مما يؤدي إلى نموذج أصغر بمقدار 5 مرات وبالتالي زيادة سرعة التدريب بشكل معتدل مع زيادة متطلبات الذاكرة بشكل معتدل أيضًا.

تتمحور معظم الأوراق البحثية والتنفيذات ذات الصلة حول "تينسورفلو/تي بي يو":

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model (GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

وبالنسبة لـ"باي تورش"، قامت "ديب سبيد" أيضًا ببناء واحد: [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596)، [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) - منشورات المدونة: [1](https://www.microsoft.com/en-us/research/blog/deepspeed-powers-8x-larger-moe-model-training-with-high-performance/)، [2](https://www.microsoft.com/en-us/research/publication/scalable-and-efficient-moe-training-for-multitask-multilingual-models/)، ونشر محدد مع نماذج توليد اللغة الطبيعية الكبيرة القائمة على المحول: [منشور المدونة](https://www.deepspeed.ai/2021/12/09/deepspeed-moe-nlg.html)، [فرع ميغاتورن-ديب سبيد](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training).

## استخدام باي تورش للاهتمام الأصلي واهتمام فلاش

يمكن أيضًا لـ "باي تورش" [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) استدعاء "فلاش أتنشن" و"كيرنلز" الاهتمام بكفاءة الذاكرة تحت الغطاء. ويجري حاليًا إضافة دعم SDPA بشكل أصلي في "ترانسفورمرز" ويتم استخدامه بشكل افتراضي لـ `torch>=2.1.1` عند توفر التنفيذ. يرجى الرجوع إلى [منتج باي تورش المُعد مسبقًا](https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) للحصول على قائمة بالنماذج المدعومة والمزيد من التفاصيل.

تفقد هذا [المنشور](https://pytorch.org/blog/out-of-the-box-acceleration/) لمعرفة المزيد حول التسريع ووفورات الذاكرة مع SDPA.