# النص أو الصورة إلى الفيديو

بفضل نجاح نماذج النشر النصي، يمكن لنماذج الفيديو التوليدية إنشاء مقاطع فيديو قصيرة بناءً على موجه نصي أو صورة أولية. وتعمل هذه النماذج على توسيع نموذج النشر المُدرب مسبقًا لإنشاء مقاطع فيديو من خلال إضافة نوع من الطبقات التلافيفية الزمنية و/أو المكانية إلى البنية. ويتم استخدام مجموعة بيانات مختلطة من الصور ومقاطع الفيديو لتدريب النموذج الذي يتعلم إخراج سلسلة من لقطات الفيديو بناءً على النص أو الصورة المشروطة.

سيوضح هذا الدليل كيفية إنشاء مقاطع فيديو، وكيفية تكوين معلمات نموذج الفيديو، وكيفية التحكم في إنشاء الفيديو.

## النماذج الشهيرة

> [!TIP]
> اكتشف نماذج أخرى رائجة وحديثة لتوليد الفيديو على [المنصة](https://huggingface.co/models?pipeline_tag=text-to-video&sort=trending)!

تعد [Stable Video Diffusions (SVD)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) و [I2VGen-XL](https://huggingface.co/ali-vilab/i2vgen-xl/) و [AnimateDiff](https://huggingface.co/guoyww/animatediff) و [ModelScopeT2V](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) نماذج شهيرة تُستخدم في نشر الفيديو. ويتميز كل نموذج بخصائص فريدة. على سبيل المثال، يقوم نموذج AnimateDiff بإدراج وحدة نمذجة الحركة في نموذج النشر النصي المجمّد لإنشاء صور متحركة مخصصة، في حين أن نموذج SVD مُدرب بالكامل من الصفر باستخدام عملية تدريب من ثلاث مراحل لإنشاء مقاطع فيديو قصيرة وعالية الجودة.

### Stable Video Diffusion

يستند نموذج [SVD](../api/pipelines/svd) إلى نموذج Stable Diffusion 2.1، وقد تم تدريبه على الصور، ثم مقاطع الفيديو منخفضة الدقة، وأخيرًا مجموعة بيانات أصغر من مقاطع الفيديو عالية الدقة. وينشئ هذا النموذج مقطع فيديو قصير يتراوح طوله بين ثانيتين وأربع ثوانٍ بناءً على صورة أولية. ويمكنك معرفة المزيد من التفاصيل حول النموذج، مثل التكييف الدقيق، في دليل [Stable Video Diffusion](../using-diffusers/svd).

ابدأ بتحميل [`StableVideoDiffusionPipeline`] وتمرير صورة أولية لإنشاء مقطع فيديو منها.

```py
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipeline = StableVideoDiffusionPipeline.from_pretrained(
"stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">مقطع الفيديو المُنشأ</figcaption>
</div>
</div>

### I2VGen-XL

نموذج [I2VGen-XL](../api/pipelines/i2vgenxl) هو نموذج نشر يمكنه إنشاء مقاطع فيديو بدقة أعلى من نموذج SVD، كما أنه قادر على قبول موجهات نصية بالإضافة إلى الصور. وقد تم تدريب النموذج باستخدام مشفرين هرميين (مُشفر التفاصيل ومُشفر شامل) لالتقاط التفاصيل منخفضة المستوى وعالية المستوى في الصور بشكل أفضل. ويتم استخدام هذه التفاصيل المُتعلمة لتدريب نموذج نشر الفيديو الذي يحسن دقة الفيديو والتفاصيل في مقطع الفيديو المُنشأ.

يمكنك استخدام نموذج I2VGen-XL من خلال تحميل [`I2VGenXLPipeline`]، وتمرير موجه نصي وصورة لإنشاء مقطع فيديو.

```py
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image = load_image(image_url).convert("RGB")

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(8888)

frames = pipeline(
prompt=prompt,
image=image,
num_inference_steps=50,
negative_prompt=negative_prompt,
guidance_scale=9.0,
generator=generator
).frames[0]
export_to_gif(frames, "i2v.gif")
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2vgen-xl-example.gif"/>
<figcaption class="mt-₂ text-center text-sm text-gray-500">مقطع الفيديو المُنشأ</figcaption>
</div>
</div>

### AnimateDiff

نموذج [AnimateDiff](../api/pipelines/animatediff) هو نموذج ملحق يقوم بإدراج وحدة نمطية للحركة في نموذج نشر مُدرب مسبقًا لإضفاء الحركة على صورة. ويتم تدريب الملحق على مقاطع الفيديو لتعلم الحركة التي تُستخدم لتهيئة عملية الإنشاء لإنشاء مقطع فيديو. ومن الأسرع والأسهل تدريب الملحق فقط، ويمكن تحميله في معظم نماذج النشر، مما يحولها بشكل فعال إلى "نماذج فيديو".

ابدأ بتحميل [`MotionAdapter`].

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
```

بعد ذلك، قم بتحميل نموذج Stable Diffusion المُدرب بشكل دقيق باستخدام [`AnimateDiffPipeline`].

```py
pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
"emilianJR/epiCRealism",
subfolder="scheduler",
clip_sample=False,
timestep_spacing="linspace",
beta_schedule="linear",
steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()
```

قم بإنشاء موجه نصي وإنشاء مقطع الفيديو.

```py
output = pipeline(
prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
negative_prompt="bad quality, worse quality, low resolution",
num_frames=16,
guidance_scale=7.5,
num_inference_steps=50,
generator=torch.Generator("cpu").manual_seed(49),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff.gif"/>
</div>

### ModelscopeT2V

يضيف نموذج [ModelscopeT2V](../api/pipelines/text_to_video) عمليات تلفيفية مكانية وزمنية واهتمام إلى شبكة UNet، وقد تم تدريبه على مجموعات بيانات النص والصورة والنص والفيديو لتحسين ما يتعلمه أثناء التدريب. ويأخذ النموذج موجهًا نصيًا، ويقوم بتشفيره وإنشاء تضمينات نصية يتم إزالة التشويش عنها بواسطة شبكة UNet، ثم فك تشفيرها بواسطة VQGAN إلى مقطع فيديو.

<Tip>
ينشئ نموذج ModelScopeT2V مقاطع فيديو عليها علامة مائية بسبب مجموعات البيانات التي تم تدريبه عليها. ولاستخدام نموذج خالٍ من العلامات المائية، جرّب نموذج [cerspense/zeroscope_v2_76w](https://huggingface.co/cerspense/zeroscope_v2_576w) مع [`TextToVideoSDPipeline`] أولاً، ثم قم بتحسين مقطع الفيديو الناتج باستخدام نقطة تفتيش [cerspense/zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL) باستخدام [`VideoToVideoSDPipeline`].
</Tip>

قم بتحميل نقطة تفتيش ModelScopeT2V في [`DiffusionPipeline`] مع موجه نصي لإنشاء مقطع فيديو.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

prompt = "Confident teddy bear surfer rides the wave in the tropics"
video_frames = pipeline(prompt).frames[0]
export_to_video(video_frames, "modelscopet2v.mp4", fps=10)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/modelscopet2v.gif" />
</div>

## تكوين معلمات النموذج

هناك بعض المعلمات المهمة التي يمكنك تكوينها في الأنبوب والتي ستؤثر على عملية إنشاء الفيديو وجودته. دعنا نلقي نظرة فاحصة على ما تفعله هذه المعلمات وكيف يؤثر تغييرها على الإخراج.
### عدد الإطارات

يحدد معامل `num_frames` عدد الإطارات التي يتم توليدها في الثانية. الإطار هو صورة يتم تشغيلها في تتابع مع إطارات أخرى لإنشاء حركة أو فيديو. يؤثر هذا على طول الفيديو لأن الأنبوب يولد عددًا معينًا من الإطارات في الثانية (تحقق من مرجع واجهة برمجة التطبيقات الافتراضية لأنبوب). لزيادة مدة الفيديو، ستحتاج إلى زيادة معامل `num_frames`.

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/num_frames_14.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">num_frames=14</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/num_frames_25.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">num_frames=25</figcaption>
</div>
</div>

### مقياس التوجيه

يتحكم معامل `guidance_scale` في مدى اتساق الفيديو المولد مع موجه النص أو الصورة الأولية. تشير قيمة `guidance_scale` أعلى إلى أن الفيديو المولد أكثر اتساقًا مع موجه النص أو الصورة الأولية، في حين أن قيمة `guidance_scale` أقل تشير إلى أن الفيديو المولد أقل اتساقًا، مما قد يمنح النموذج مزيدًا من "الإبداع" لتفسير إدخال التكييف.

<Tip>
يستخدم SVD معاملات `min_guidance_scale` و`max_guidance_scale` لتطبيق التوجيه على الإطارات الأولى والأخيرة على التوالي.
</Tip>

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2vgen-xl-example.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale=9.0</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guidance_scale_1.0.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale=1.0</figcaption>
</div>
</div>

### موجه سلبي

يثبط الموجه السلبي النموذج عن توليد أشياء لا تريدها. ويُستخدم هذا المعامل بشكل شائع لتحسين جودة التوليد بشكل عام عن طريق إزالة الميزات السيئة أو السيئة مثل "دقة منخفضة" أو "تفاصيل سيئة".

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff_no_neg.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">بدون موجه سلبي</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff_neg.gif"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تم تطبيق موجه سلبي</figcaption>
</div>
</div>

### معاملات خاصة بالنموذج

هناك بعض معاملات الأنابيب التي تكون فريدة من نوعها لكل نموذج، مثل ضبط الحركة في الفيديو أو إضافة ضوضاء إلى الصورة الأولية.

<hfoptions id="special-parameters">
<hfoption id="Stable Video Diffusion">
يوفر Stable Video Diffusion تكييفًا دقيقًا إضافيًا لمعدل الإطارات بمعامل `fps` وللحركة بمعامل `motion_bucket_id`. معًا، تسمح هذه المعاملات بتعديل مقدار الحركة في الفيديو المولد.

هناك أيضًا معامل `noise_aug_strength` الذي يزيد من مقدار الضوضاء المضافة إلى الصورة الأولية. يؤثر تغيير هذا المعامل على مدى تشابه الفيديو المولد والصورة الأولية. كما تزيد قيمة `noise_aug_strength` الأعلى من مقدار الحركة. لمزيد من المعلومات، اقرأ دليل [التكييف الدقيق](../using-diffusers/svd#micro-conditioning).
</hfoption>
<hfoption id="Text2Video-Zero">
يحسب Text2Video-Zero مقدار الحركة التي يجب تطبيقها على كل إطار من اللانتماءات العشوائية. يمكنك استخدام معاملات `motion_field_strength_x` و`motion_field_strength_y` للتحكم في مقدار الحركة التي يجب تطبيقها على محوري x وy للفيديو. المعاملات `t0` و`t1` هي الخطوات الزمنية لتطبيق الحركة على اللانتماءات.
</hfoption>
</hfoptions>

## التحكم في توليد الفيديو

يمكن التحكم في توليد الفيديو بشكل مشابه للطريقة التي يتم بها التحكم في النص إلى الصورة والصورة إلى الصورة والطلاء، وذلك باستخدام [`ControlNetModel`]. الفرق الوحيد هو أنك بحاجة إلى استخدام [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] حتى يحضر كل إطار الإطار الأول.
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين:

### Text2Video-Zero
يمكن أن يعتمد Text2Video-Zero لتوليد الفيديو على صور الوضع والحافة لمزيد من التحكم في حركة الموضوع في الفيديو المولد أو للحفاظ على هوية موضوع/كائن في الفيديو. يمكنك أيضًا استخدام Text2Video-Zero مع InstructPix2Pix لتحرير مقاطع الفيديو باستخدام النص.

#### التحكم في الوضع
ابدأ بتنزيل مقطع فيديو واستخراج صور الوضع منه.

قم بتحميل [`ControlNetModel`] لتقدير الوضع ونقطة تفتيش في [`StableDiffusionControlNetPipeline`]. بعد ذلك، ستستخدم [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] لـ UNet و ControlNet.

قم بتثبيت latents لجميع الإطارات، ثم قم بتمرير موجهك وصور الوضع المستخرجة إلى النموذج لتوليد مقطع فيديو.

#### التحكم في الحافة
قم بتنزيل مقطع فيديو واستخراج الحواف منه.

قم بتحميل [`ControlNetModel`] للحافة الحادة ونقطة تفتيش في [`StableDiffusionControlNetPipeline`]. بعد ذلك، ستستخدم [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] لـ UNet و ControlNet.

قم بتثبيت latents لجميع الإطارات، ثم قم بتمرير موجهك وصور الحافة المستخرجة إلى النموذج لتوليد مقطع فيديو.

#### InstructPix2Pix
يسمح لك InstructPix2Pix باستخدام النص لوصف التغييرات التي تريد إجراؤها على الفيديو. ابدأ بتنزيل مقطع فيديو وقراءته.

قم بتحميل [`StableDiffusionInstructPix2PixPipeline`] وتعيين [`~pipelines.text_to_video_synthesis.pipeline_text_to_video_zero.CrossFrameAttnProcessor`] لـ UNet.

مرر موجهًا يصف التغيير الذي تريد تطبيقه على الفيديو.

## تحسين
يتطلب إنشاء الفيديو الكثير من الذاكرة لأنك تقوم بتوليد العديد من إطارات الفيديو مرة واحدة. يمكنك تقليل متطلبات الذاكرة على حساب بعض سرعة الاستدلال. جرّب ما يلي:

1. نقل مكونات الأنابيب التي لم تعد هناك حاجة إليها إلى وحدة المعالجة المركزية
2. يقوم التقطيع التغذوي الأمامي بتشغيل طبقة التغذية الأمامية في حلقة بدلاً من تشغيلها جميعًا مرة واحدة
3. قم بتقسيم عدد الإطارات التي يجب على VAE فك تشفيرها إلى مجموعات بدلاً من فك تشفيرها جميعًا مرة واحدة

إذا لم تكن الذاكرة مشكلة وتريد تحسين السرعة، فجرّب لف UNet مع [`torch.compile`](../optimization/torch2.0#torchcompile).