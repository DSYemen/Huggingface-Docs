# فهم الأنابيب والنماذج وجدولة المواعيد

[[open-in-colab]]

🧨 صُممت Diffusers لتكون صندوق أدوات سهل الاستخدام ومرن لبناء أنظمة الانتشار المخصصة لحالتك الاستخدامية. ويعد النموذجان وجدولة المواعيد هما جوهر صندوق الأدوات. في حين أن [`DiffusionPipeline`] يجمع بين هذه المكونات معًا للراحة، يمكنك أيضًا فك حزم الأنبوب واستخدام النماذج وجدولة المواعيد بشكل منفصل لإنشاء أنظمة انتشار جديدة.

في هذا البرنامج التعليمي، ستتعلم كيفية استخدام النماذج وجدولة المواعيد لتجميع نظام انتشار للاستدلال، بدءًا من خط أنابيب أساسي ثم الانتقال إلى خط أنابيب Stable Diffusion.

## فك خط أنابيب أساسي

يعد خط الأنابيب طريقة سريعة وسهلة لتشغيل نموذج للاستدلال، ولا يتطلب أكثر من أربعة أسطر من التعليمات البرمجية لتوليد صورة:

```py
>>> from diffusers import DDPMPipeline

>>> ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
>>> image = ddpm(num_inference_steps=25).images[0]
>>> image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ddpm-cat.png" alt="صورة قطة تم إنشاؤها من DDPMPipeline" />
</div>

كان ذلك سهلاً للغاية، ولكن كيف قام خط الأنابيب بذلك؟ دعونا نقوم بتفكيك خط الأنابيب ونلقي نظرة على ما يحدث خلف الكواليس.

في المثال أعلاه، يحتوي خط الأنابيب على نموذج [`UNet2DModel`] ومخطط [`DDPMScheduler`]. يقوم خط الأنابيب بإزالة ضوضاء الصورة عن طريق أخذ ضوضاء عشوائية بحجم الإخراج المطلوب ومروره عبر النموذج عدة مرات. في كل خطوة زمنية، يتنبأ النموذج ب *بقايا الضوضاء* ويستخدمها الجدول الزمني للتنبؤ بصورة أقل ضوضاء. يكرر خط الأنابيب هذه العملية حتى يصل إلى نهاية عدد خطوات الاستدلال المحددة.

لإعادة إنشاء خط الأنابيب مع النموذج والجدول الزمني بشكل منفصل، دعونا نكتب عملية إزالة الضوضاء الخاصة بنا.

1. قم بتحميل النموذج والجدول الزمني:

```py
>>> from diffusers import DDPMScheduler, UNet2DModel

>>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
>>> model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
```

2. قم بتعيين عدد الخطوات الزمنية لتشغيل عملية إزالة الضوضاء:

```py
>>> scheduler.set_timesteps(50)
```

3. يؤدي تعيين الخطوات الزمنية للجدول الزمني إلى إنشاء مصفوفة ذات عناصر متباعدة بالتساوي، 50 في هذا المثال. يقابل كل عنصر خطوة زمنية يقوم فيها النموذج بإزالة ضوضاء الصورة. عندما تقوم بإنشاء حلقة إزالة الضوضاء لاحقًا، فستقوم بالتعيين على هذه المصفوفة لإزالة ضوضاء الصورة:

```py
>>> scheduler.timesteps
tensor([980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720,
700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440,
420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160,
140, 120, 100,  80,  60,  40,  20,   0])
```

4. قم بإنشاء بعض الضوضاء العشوائية بنفس شكل الإخراج المطلوب:

```py
>>> import torch

>>> sample_size = model.config.sample_size
>>> noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
```

5. اكتب الآن حلقة للتعيين على الخطوات الزمنية. في كل خطوة زمنية، يقوم النموذج بتنفيذ تمريرة [`UNet2DModel.forward`] ويعيد بقايا الضوضاء. تستخدم طريقة [`~DDPMScheduler.step`] الخاصة بالجدول الزمني بقايا الضوضاء والخطوة الزمنية والإدخال للتنبؤ بالصورة في الخطوة الزمنية السابقة. يصبح هذا الإخراج الإدخال التالي للنموذج في حلقة إزالة الضوضاء، وسوف يتكرر حتى يصل إلى نهاية صفيف "الخطوات الزمنية".

```py
>>> input = noise

>>> for t in scheduler.timesteps:
...     with torch.no_grad():
...         noisy_residual = model(input, t).sample
...     previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
...     input = previous_noisy_sample
```

هذه هي عملية إزالة الضوضاء بأكملها، ويمكنك استخدام هذا النمط نفسه لكتابة أي نظام انتشار.

6. الخطوة الأخيرة هي تحويل الإخراج المزال ضوضاؤه إلى صورة:

```py
>>> from PIL import Image
>>> import numpy as np

>>> image = (input / 2 + 0.5).clamp(0, 1).squeeze()
>>> image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
>>> image = Image.fromarray(image)
>>> image
```

في القسم التالي، ستختبر مهاراتك عن طريق تفكيك خط أنابيب Stable Diffusion الأكثر تعقيدًا. الخطوات هي نفسها تقريبًا. ستقوم بتهيئة المكونات اللازمة، وتعيين عدد الخطوات الزمنية لإنشاء صفيف "الخطوات الزمنية". يتم استخدام صفيف "الخطوات الزمنية" في حلقة إزالة الضوضاء، وفي كل عنصر من هذا الصفيف، يتنبأ النموذج بصورة أقل ضوضاء. تتعيين حلقة إزالة الضوضاء على "الخطوات الزمنية"، وفي كل خطوة زمنية، تقوم بإخراج بقايا الضوضاء ويستخدمها الجدول الزمني للتنبؤ بصورة أقل ضوضاء في الخطوة الزمنية السابقة. تتكرر هذه العملية حتى تصل إلى نهاية صفيف "الخطوات الزمنية".

دعونا نجرب ذلك!

## فك خط أنابيب Stable Diffusion

Stable Diffusion هو نموذج انتشار نصي إلى صورة *latent*. يطلق عليه نموذج انتشار خفي لأنه يعمل مع تمثيل منخفض الأبعاد للصورة بدلاً من مساحة البكسل الفعلية، مما يجعله أكثر كفاءة في الذاكرة. يقوم المشفر بضغط الصورة إلى تمثيل أصغر، ويتم استخدام فك تشفير لتحويل التمثيل المضغوط مرة أخرى إلى صورة. بالنسبة لنماذج النص إلى الصورة، ستحتاج إلى محدد موضع ومشفر لإنشاء تضمينات نصية. من المثال السابق، أنت تعرف بالفعل أنك بحاجة إلى نموذج UNet ومخطط.

كما ترون، هذا أكثر تعقيدًا بالفعل من خط أنابيب DDPM الذي يحتوي فقط على نموذج UNet. يحتوي نموذج Stable Diffusion على ثلاثة نماذج منفصلة مسبقًا.

<Tip>
💡 اقرأ مدونة [كيف يعمل Stable Diffusion؟](Https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) لمزيد من التفاصيل حول كيفية عمل نماذج VAE وUNet ومشفر النص.
</Tip>

الآن بعد أن عرفت ما تحتاجه لخط أنابيب Stable Diffusion، قم بتحميل جميع هذه المكونات باستخدام طريقة [`~ModelMixin.from_pretrained`]. يمكنك العثور عليها في نقطة التحقق المسبقة [`runwayml/stable-diffusion-v1-5`] (https://huggingface.co/runwayml/stable-diffusion-v1-5)، ويتم تخزين كل مكون في مجلد فرعي منفصل:

```py
>>> from PIL import Image
>>> import torch
>>> from transformers import CLIPTextModel، CLIPTokenizer
>>> from diffusers import AutoencoderKL، UNet2DConditionModel، PNDMScheduler

>>> vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4"، subfolder="vae"، use_safetensors=True)
>>> tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4"، subfolder="tokenizer")
>>> text_encoder = CLIPTextModel.from_pretrained(
...     "CompVis/stable-diffusion-v1-4"، subfolder="text_encoder"، use_safetensors=True
... )
>>> unet = UNet2DConditionModel.from_pretrained(
...     "CompVis/stable-diffusion-v1-4"، subfolder="unet"، use_safetensors=True
... )
```

بدلاً من جدول [`PNDMScheduler`] الافتراضي، استبدله بجدول [`UniPCMultistepScheduler`] لمعرفة مدى سهولة توصيل جدول زمني مختلف:

```py
>>> from diffusers import UniPCMultistepScheduler

>>> scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4"، subfolder="scheduler")
```

للإسراع في الاستدلال، قم بنقل النماذج إلى وحدة معالجة الرسومات (GPU) لأنها تحتوي على أوزان قابلة للتدريب، على عكس الجدول الزمني:

```py
>>> torch_device = "cuda"
>>> vae.to(torch_device)
>>> text_encoder.to(torch_device)
>>> unet.to(torch_device)
```

### إنشاء تضمينات نصية

الخطوة التالية هي توكينز النص لتوليد التضمينات. يتم استخدام النص لتكييف نموذج UNet وتوجيه عملية الانتشار نحو شيء يشبه موجه النص.

<Tip>
💡 يحدد معلمة "guidance_scale" مقدار الوزن الذي يجب إعطاؤه للإشارة عند إنشاء صورة.
</Tip>

لا تتردد في اختيار أي موجه تريد إذا كنت تريد إنشاء شيء آخر!

```py
>>> prompt = ["صورة لرواد الفضاء يركبون حصانًا"]
>>> height = 512 # الارتفاع الافتراضي لـ Stable Diffusion
>>> width = 512 # العرض الافتراضي لـ Stable Diffusion
>>> num_inference_steps = 25 # عدد خطوات إزالة الضوضاء
>>> guidance_scale = 7.5 # مقياس التوجيه الخالي من التصنيف
>>> generator = torch.manual_seed(0) # بذرة مولد لإنشاء الضوضاء الأولية
>>> batch_size = len(prompt)
```

قم بتوكينز النص وإنشاء التضمينات من الموجه:

```py
>>> text_input = tokenizer(
...     prompt، padding="max_length"، max_length=tokenizer.model_max_length، truncation=True، return_tensors="pt"
... )

>>> with torch.no_grad():
...     text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```

ستحتاج أيضًا إلى إنشاء *تضمينات النص غير المشروطة* وهي التضمينات الخاصة برمز الحشو. يجب أن يكون لهذه التضمينات نفس الشكل (`batch_size` و`seq_length`) مثل التضمينات المشروطة `text_embeddings`:

```py
>>> max_length = text_input.input_ids.shape[-1]
>>> uncond_input = tokenizer([""] * batch_size، padding="max_length"، max_length=max_length، return_tensors="pt")
>>> uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
```

دعونا نقوم بدمج التضمينات المشروطة وغير المشروطة في دفعة لتجنب إجراء تمريرين:

```py
>>> text_embeddings = torch.cat ([uncond_embeddings، text_embeddings])
```

### إنشاء ضوضاء عشوائية

بعد ذلك، قم بتوليد بعض الضوضاء العشوائية كنقطة بداية لعملية الانتشار. هذا هو التمثيل الخفي للصورة، وسيتم إزالة ضوضاؤه تدريجياً. في هذه المرحلة، الصورة "الكامنة" أصغر من أبعاد الصورة النهائية ولكن هذا أمر طبيعي لأن النموذج سيحولها إلى أبعاد الصورة النهائية 512x512 لاحقًا.

<Tip>
💡 يتم تقسيم الارتفاع والعرض على 8 لأن نموذج "vae" يحتوي على 3 طبقات تقليل التدرج. يمكنك التحقق من ذلك عن طريق تشغيل ما يلي:
```py
2 ** (len (vae.config.block_out_channels) - 1) == 8
```
</Tip>

```py
>>> latents = torch.randn(
...     (batch_size، unet.config.in_channels، height // 8، width // 8)،
...     generator=generator،
...     device=torch_device،
... )
```

### إزالة ضوضاء الصورة

ابدأ عن طريق قياس الإدخال بتوزيع الضوضاء الأولية، *sigma*، وقيمة مقياس الضوضاء، والتي تكون مطلوبة لجدولة المواعيد المحسنة مثل [`UniPCMultistepScheduler`]:

```py
>>> latents = latents * scheduler.init_noise_sigma
```

الخطوة الأخيرة هي إنشاء حلقة إزالة الضوضاء التي ستحول الضوضاء البحتة في "latents" إلى صورة موصوفة بموجهك. تذكر، تحتاج حلقة إزالة الضوضاء إلى القيام بثلاثة أشياء:

1. قم بتعيين الخطوات الزمنية للجدول الزمني لاستخدامها أثناء إزالة الضوضاء.
2. التعيين على الخطوات الزمنية.
3. في كل خطوة زمنية، اتصل بنموذج UNet للتنبؤ ببقايا الضوضاء ومررها إلى الجدول الزمني لحساب عينة الضوضاء السابقة.

```py
>>> from tqdm.auto import tqdm

>>> scheduler.set_timesteps(num_inference_steps)

>>> for t in tqdm(scheduler.timesteps):
...     # قم بتوسيع latents إذا كنا نقوم بالتوجيه الخالي من التصنيف لتجنب إجراء تمريرين.
...     latent_model_input = torch.cat ([latents] * 2)

...     latent_model_input = scheduler.scale_model_input (latent_model_input، timestep=t)

...     # التنبؤ ببقايا الضوضاء
...     with torch.no_grad():
...         noise_pred = unet (latent_model_input، t، encoder_hidden_states=text_embeddings).sample

...     # أداء التوجيه
...     noise_pred_uncond، noise_pred_text = noise_pred.chunk (2)
...     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

...     # احسب عينة الضوضاء السابقة x_t -> x_t-1
...     latents = scheduler.step (noise_pred، t، latents).prev_sample
```

### فك تشفير الصورة

الخطوة الأخيرة هي استخدام "vae" لفك تشفير التمثيل الخفي إلى صورة والحصول على الإخراج المفكك باستخدام "sample":

```py
# قسّم وشفّر صورة latents باستخدام vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
image = vae.decode (latents).sample
```

أخيرًا، قم بتحويل الصورة إلى `PIL.Image` لم
## الخطوات التالية
من الأنابيب الأساسية إلى المعقدة، رأيت أن كل ما تحتاجه حقًا لكتابة نظام الانتشار الخاص بك هو حلقة إزالة التشويش. يجب أن تقوم الحلقة بضبط خطوات الوقت للمجدول، والتكرار فوقها، والتبديل بين استدعاء نموذج UNet للتنبؤ ببقايا الضوضاء ونقلها إلى المجدول لحساب العينة المشوشة السابقة.
هذا هو حقًا ما تم تصميم 🧨 Diffusers من أجله: لجعل كتابة نظام الانتشار الخاص بك بديهية وسهلة باستخدام النماذج والمجدولات.
في خطواتك التالية، لا تتردد في:
* تعلم كيفية [بناء والمساهمة بخط أنابيب](../using-diffusers/contribute_pipeline) إلى 🧨 Diffusers. لا يمكننا الانتظار لمعرفة ما ستتوصل إليه!
* استكشف [خطوط الأنابيب الموجودة](../api/pipelines/overview) في المكتبة، وانظر ما إذا كان بإمكانك تفكيك وبناء خط أنابيب من الصفر باستخدام النماذج والمجدولات بشكل منفصل.