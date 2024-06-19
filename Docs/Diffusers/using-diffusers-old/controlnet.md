# ControlNet

ControlNet هو نوع من أنواع النماذج المستخدمة للتحكم في نماذج انتشار الصور من خلال توفير دخل إضافي للصورة. هناك العديد من أنواع مدخلات التحكم (Canny edge، رسم المستخدم، وضع الإنسان، العمق، والمزيد) التي يمكنك استخدامها للتحكم في نموذج الانتشار. وهذا مفيد للغاية لأنه يمنحك تحكمًا أكبر في إنشاء الصور، مما يسهل إنشاء صور محددة دون الحاجة إلى تجربة مطالبات نصية أو قيم إزالة الضوضاء المختلفة.

للاطلاع على قائمة بتنفيذات ControlNet لمختلف مدخلات التحكم، راجع القسم 3.5 من ورقة ControlNet v1. يمكنك العثور على نماذج ControlNet الرسمية المشروطة على ملف تعريف Hub الخاص بـ [lllyasviel]، والمزيد من النماذج [المدربة من قبل المجتمع] على Hub.

بالنسبة لنماذج ControlNet Stable Diffusion XL (SDXL)، يمكنك العثور عليها في منظمة 🤗 [Diffusers] Hub، أو يمكنك تصفح النماذج [المدربة من قبل المجتمع] على Hub.

يحتوي نموذج ControlNet على مجموعتين من الأوزان (أو الكتل) متصلة بطبقة التصفية الصفرية:

- *نسخة محمية* تحافظ على كل ما تعلمه نموذج الانتشار المُدرب مسبقًا
- *نسخة قابلة للتدريب* يتم تدريبها على إدخال التحكم الإضافي

نظرًا لأن النسخة المحمية تحافظ على النموذج المُدرب مسبقًا، فإن تدريب وتنفيذ ControlNet على إدخال تحكم جديد سريع مثل ضبط نموذج آخر لأنه لا يتم تدريب النموذج من الصفر.

سيوضح هذا الدليل كيفية استخدام ControlNet للتحويل من نص إلى صورة، ومن صورة إلى صورة، والتلوين، والمزيد! هناك العديد من أنواع مدخلات التحكم في ControlNet للاختيار من بينها، ولكن في هذا الدليل، سنركز فقط على بعض منها. لا تتردد في تجربة مدخلات التحكم الأخرى!

قبل البدء، تأكد من تثبيت المكتبات التالية:

## من نص إلى صورة

بالنسبة للنص إلى الصورة، عادةً ما يتم تمرير موجه نصي إلى النموذج. ولكن مع ControlNet، يمكنك تحديد إدخال تحكم إضافي. دعونا نشترط على النموذج باستخدام صورة Canny، وهو مخطط أبيض لصورة على خلفية سوداء. بهذه الطريقة، يمكن لـ ControlNet استخدام صورة Canny كعنصر تحكم لتوجيه النموذج لإنشاء صورة لها نفس المخطط.

قم بتحميل صورة واستخدم مكتبة [opencv-python] لاستخراج صورة Canny:

تظهر الصورتان التاليتان الصورة الأصلية وصورة Canny:

بعد ذلك، قم بتحميل نموذج ControlNet المشروط بالكشف عن حافة Canny ومرره إلى [`StableDiffusionControlNetPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين نقل النموذج إلى وحدة المعالجة المركزية للتسريع وتقليل استخدام الذاكرة.

الآن قم بتمرير موجه النص وصورة Canny إلى الأنبوب:

## من صورة إلى صورة

بالنسبة للصور إلى الصور، عادةً ما يتم تمرير صورة أولية وموجه إلى الأنبوب لإنشاء صورة جديدة. مع ControlNet، يمكنك تمرير إدخال تحكم إضافي لتوجيه النموذج. دعونا نشترط على النموذج باستخدام خريطة العمق، وهي صورة تحتوي على معلومات مكانية. بهذه الطريقة، يمكن لـ ControlNet استخدام خريطة العمق كعنصر تحكم لتوجيه النموذج لإنشاء صورة تحافظ على المعلومات المكانية.

ستستخدم [`StableDiffusionControlNetImg2ImgPipeline`] لهذه المهمة، والتي تختلف عن [`StableDiffusionControlNetPipeline`] لأنها تسمح لك بتمرير صورة أولية كنقطة بداية لعملية إنشاء الصورة.

قم بتحميل صورة واستخدم خط أنابيب "التقدير العميق" [`~transformers`] من 🤗 Transformers لاستخراج خريطة عمق الصورة:

بعد ذلك، قم بتحميل نموذج ControlNet المشروط بخرائط العمق ومرره إلى [`StableDiffusionControlNetImg2ImgPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين نقل النموذج إلى وحدة المعالجة المركزية للتسريع وتقليل استخدام الذاكرة.

الآن قم بتمرير موجه النص والصورة الأولية وخريطة العمق إلى الأنبوب:

تظهر الصورتان التاليتان الصورة الأصلية والصورة التي تم إنشاؤها:
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع اتباع التعليمات التي قدمتها.

## إكمال الصور (Inpainting)

بالنسبة لإكمال الصور، فأنت بحاجة إلى صورة أولية، وصورة قناع، ووصف يوضح ما يجب استبدال القناع به. تسمح نماذج ControlNet بإضافة صورة تحكم أخرى لتهيئة النموذج. دعنا نستخدم صورة القناع لإكمال الصورة لتهيئة النموذج. بهذه الطريقة، يمكن لشبكة ControlNet استخدام صورة القناع كتحكم لتوجيه النموذج لتوليد صورة داخل منطقة القناع.

قم بتحميل صورة أولية وصورة قناع:

قم بإنشاء دالة لإعداد صورة التحكم من الصورة الأولية وصورة القناع. سيؤدي هذا إلى إنشاء مصفوفة لتحديد البكسلات في الصورة الأولية كقناع إذا كان البكسل المقابل في صورة القناع أعلى من عتبة معينة.

## وضع التخمين (Guess mode)

[وضع التخمين](https://github.com/lllyasviel/ControlNet/discussions/188) لا يتطلب تقديم موجه على الإطلاق إلى ControlNet! وهذا يجبر مشفر ControlNet على بذل قصارى جهده "لتخمين" محتويات خريطة التحكم المدخلة (خريطة العمق، أو تقدير الوضع، أو حواف كاني، وما إلى ذلك).

يعدل وضع التخمين مقياس المخلفات الناتجة عن ControlNet وفقًا لنسبة ثابتة تعتمد على عمق الكتلة. يقابل الكتلة الأعمق نسبة 0.1، ومع ازدياد عمق الكتل، يزداد المقياس بشكل أسّي بحيث يصبح مقياس إخراج كتلة MidBlock 1.0.

<Tip>

لا يؤثر وضع التخمين على توجيه الموجه، ويمكنك تقديم موجه إذا أردت ذلك.

</Tip>

قم بتعيين `guess_mode=True` في خط الأنابيب، ومن [المستحسن](https://github.com/lllyasviel/ControlNet#guess-mode--non-prompt-mode) تعيين قيمة `guidance_scale` بين 3.0 و5.0.
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين فقط مع مراعاة التعليمات التي قدمتها.

## ControlNet مع Stable Diffusion XL

في الوقت الحالي، لا يوجد الكثير من نماذج ControlNet المتوافقة مع Stable Diffusion XL (SDXL)، ولكننا قمنا بتدريب نموذجين كاملين من ControlNet لـ SDXL المشروطة على كشف حافة كاني وخرائط العمق. كما نجري تجارب لإنشاء إصدارات أصغر من نماذج ControlNet المتوافقة مع SDXL بحيث يسهل تشغيلها على الأجهزة ذات الموارد المحدودة. يمكنك العثور على هذه النقاط المرجعية على [منظمة 🤗 Diffusers Hub](https://huggingface.co/diffusers)!

دعونا نستخدم SDXL ControlNet المشروط على صور كاني لتوليد صورة. ابدأ بتحميل صورة وإعداد صورة كاني:

بعد ذلك، قم بتحميل نموذج SDXL ControlNet المشروط على كشف حافة كاني ومرره إلى [`StableDiffusionXLControlNetPipeline`]. يمكنك أيضًا تمكين نقل النموذج إلى وحدة المعالجة المركزية لتقليل استخدام الذاكرة.

الآن، قم بتمرير موجهك (واختياريًا، موجه سلبي إذا كنت تستخدم واحدًا) وصورة كاني إلى الأنبوب:

<Tip>

تحدد وسيطة [`controlnet_conditioning_scale`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale) مقدار الوزن الذي يجب تعيينه لمدخلات المشروط. القيمة الموصى بها هي 0.5 للتعميم الجيد، ولكن لا تتردد في التجربة بهذا الرقم!

</Tip>

يمكنك أيضًا استخدام [`StableDiffusionXLControlNetPipeline`] في وضع التخمين عن طريق تعيين المعلمة إلى `True`:

<Tip>

يمكنك استخدام نموذج المُحسِّن مع `StableDiffusionXLControlNetPipeline` لتحسين جودة الصورة، تمامًا كما يمكنك ذلك مع `StableDiffusionXLPipeline` العادي.

راجع قسم [تحسين جودة الصورة](./sdxl#refine-image-quality) لمعرفة كيفية استخدام نموذج المُحسِّن.

تأكد من استخدام `StableDiffusionXLControlNetPipeline` ومرر `image` و `controlnet_conditioning_scale`.

```py
base = StableDiffusionXLControlNetPipeline(...)
image = base(
prompt=prompt,
controlnet_conditioning_scale=0.5,
image=canny_image,
num_inference_steps=40,
denoising_end=0.8,
output_type="latent",
).images
# الباقي كما هو تمامًا مع StableDiffusionXLPipeline
```

</Tip>
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع اتباع التعليمات التي قدمتها.

## MultiControlNet

يمكنك تكوين العديد من عمليات ضبط ControlNet من مدخلات الصور المختلفة لإنشاء *MultiControlNet*. وللحصول على نتائج أفضل، من المفيد غالبًا:

1. قناع الضبط بحيث لا تتداخل (على سبيل المثال، قناع منطقة صورة Canny حيث يقع ضبط الوضع)
2. تجربة مع [`controlnet_conditioning_scale`] (https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale) لتحديد مقدار الوزن الذي يجب تعيينه لكل إدخال ضبط

في هذا المثال، ستقوم بدمج صورة Canny وصورة تقدير وضع الإنسان لإنشاء صورة جديدة.

قم بإعداد ضبط صورة Canny:

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/landscape_canny_masked.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة Canny</figcaption>
</div>
</div>

بالنسبة لتقدير الوضع البشري، قم بتثبيت [controlnet_aux] (https://github.com/patrickvonplaten/controlnet_aux):

قم بإعداد ضبط تقدير الوضع البشري:

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/person_pose.png"/>
<figcaption class="mt-min text-center text-sm text-gray-500">صورة الوضع البشري</figcaption>
</div>
</div>

قم بتحميل قائمة نماذج ControlNet التي تتوافق مع كل ضبط، ومررها إلى [`StableDiffusionXLControlNetPipeline`]. استخدم [`UniPCMultistepScheduler`] الأسرع وقم بتمكين تفريغ النموذج لتقليل استخدام الذاكرة.

الآن يمكنك تمرير موجهك (سالبًا إذا كنت تستخدم واحدًا)، وصورة Canny، وصورة الوضع إلى الأنبوب:

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multicontrolnet.png"/>
</div>