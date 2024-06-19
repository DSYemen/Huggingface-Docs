# كاندينسكي

تعد نماذج كاندينسكي سلسلة من النماذج متعددة اللغات لتوليد الصور النصية. يستخدم نموذج Kandinsky 2.0 مشفرين نصيين متعددي اللغات ويقوم بتجميع تلك النتائج لـ UNet.

يغير Kandinsky 2.1 البنية لتضمين نموذج الصورة الأولية ([` CLIP `](https://huggingface.co/docs/transformers/model_doc/clip)) لتوليد خريطة بين النصوص وتضمين الصور. توفر الخريطة محاذاة أفضل للنص والصورة ويتم استخدامها مع تضمين النص أثناء التدريب، مما يؤدي إلى نتائج عالية الجودة. وأخيرًا، يستخدم Kandinsky 2.1 [فك تشفير Modulating Quantized Vectors (MoVQ)](https://huggingface.co/papers/2209.09002) - والذي يضيف طبقة التطبيع الشرطي المكاني لزيادة الواقعية الفوتوغرافية - لترميز latents إلى صور.

يحسن Kandinsky 2.2 النموذج السابق عن طريق استبدال مشفر الصورة في نموذج الصورة الأولية بنموذج CLIP-ViT-G أكبر لتحسين الجودة. كما أعيد تدريب نموذج الصورة الأولية على صور ذات دقات وجوانب مختلفة لتوليد صور عالية الدقة وأحجام صور مختلفة.

يبسط Kandinsky 3 البنية وينتقل بعيدًا عن عملية التوليد ذات المرحلتين التي تتضمن النموذج الأولي ونموذج الانتشار. بدلاً من ذلك، يستخدم Kandinsky 3 [Flan-UL2](https://huggingface.co/google/flan-ul2) لتشفير النص، و UNet مع [BigGan-deep](https://hf.co/papers/1809.11096) blocks، و [Sber-MoVQGAN](https://github.com/ai-forever/MoVQGAN) لترميز latents إلى صور. يتم تحقيق فهم النص وجودة الصورة المولدة بشكل أساسي من خلال استخدام مشفر نصي أكبر و UNet.

سيوضح هذا الدليل كيفية استخدام نماذج Kandinsky للصورة النصية، والصورة إلى صورة، والتلوين، والتنقل، والمزيد.

قبل البدء، تأكد من تثبيت المكتبات التالية:

<Tip warning={true}>

يعد استخدام Kandinsky 2.1 و 2.2 متشابهًا جدًا! الفرق الوحيد هو أن Kandinsky 2.2 لا يقبل `prompt` كإدخال عند فك تشفير latents. بدلاً من ذلك، يقبل Kandinsky 2.2 فقط `image_embeds` أثناء فك التشفير.

<br>

لدى Kandinsky 3 بنية أكثر إحكاما ولا يتطلب نموذجًا أوليًا. وهذا يعني أن استخدامه مطابق لنماذج الانتشار الأخرى مثل [Stable Diffusion XL](sdxl).

</Tip>

## نص إلى صورة

لاستخدام نماذج Kandinsky لأي مهمة، ابدأ دائمًا بإعداد خط أنابيب النموذج الأولي لتشفير الفكرة وتوليد تضمين الصورة. كما يقوم خط أنابيب النموذج الأولي بتوليد `negative_image_embeds` التي تتوافق مع الفكرة السلبية `""`. للحصول على نتائج أفضل، يمكنك تمرير `negative_prompt` الفعلي إلى خط أنابيب النموذج الأولي، ولكن هذا سيزيد من حجم الدفعة الفعالة لخط أنابيب النموذج الأولي بمقدار مرتين.

<hfoptions id="text-to-image">

<hfoption id="Kandinsky 2.1">

```py
from diffusers import KandinskyPriorPipeline, KandinskyPipeline
import torch

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16).to("cuda")
pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16).to("cuda")

prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
negative_prompt = "low quality, bad quality" # اختياري لتضمين فكرة سلبية، ولكن النتائج تكون أفضل عادةً
image_embeds، negative_image_embeds = prior_pipeline(prompt، negative_prompt، guidance_scale=1.0).to_tuple()
```

الآن قم بتمرير جميع المطالبات والتضمينات إلى [`KandinskyPipeline`] لتوليد صورة:

```py
image = pipeline(prompt، image_embeds=image_embeds، negative_prompt=negative_prompt، negative_image_embeds=negative_image_embeds، height=768، width=768).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-docs/cheeseburger.png"/>
</div>

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
import torch

prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior"، torch_dtype=torch.float16).to("cuda")
pipeline = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder"، torch_dtype=torch.float16).to("cuda")

prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
negative_prompt = "low quality, bad quality" # اختياري لتضمين فكرة سلبية، ولكن النتائج تكون أفضل عادةً
image_embeds، negative_image_embeds = prior_pipeline(prompt، guidance_scale=1.0).to_tuple()
```

مرر `image_embeds` و` negative_image_embeds` إلى [`KandinskyV22Pipeline`] لتوليد صورة:

```py
image = pipeline(image_embeds=image_embeds، negative_image_embeds=negative_image_embeds، height=768، width=768).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-text-to-image.png"/>
</div>

</hfoption>

<hfoption id="Kandinsky 3">

Kandinsky 3 لا يتطلب نموذجًا أوليًا، لذا يمكنك تحميل [`Kandinsky3Pipeline`] مباشرةً وتمرير فكرة لتوليد صورة:

```py
from diffusers import Kandinsky3Pipeline
import torch

pipeline = Kandinsky3Pipeline.from_pretrained("kandinsky-community/kandinsky-3"، variant="fp16"، torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
image = pipeline(prompt).images[0]
image
```

</hfoption>

</hfoptions>

يوفر 🤗 Diffusers أيضًا واجهة برمجة تطبيقات (API) شاملة مع [`KandinskyCombinedPipeline`] و [`KandinskyV22CombinedPipeline`]، مما يعني أنه لا يلزم تحميل خط أنابيب النموذج الأولي وخط أنابيب النص إلى الصورة بشكل منفصل. يقوم خط الأنابيب المجمع بتحميل كل من النموذج الأولي وفك التشفير تلقائيًا. لا يزال بإمكانك تعيين قيم مختلفة لخط أنابيب النموذج الأولي باستخدام معلمات `prior_guidance_scale` و` prior_num_inference_steps` إذا أردت.

استخدم [`AutoPipelineForText2Image`] لاستدعاء خطوط الأنابيب المجمعة تلقائيًا:

<hfoptions id="text-to-image">

<hfoption id="Kandinsky 2.1">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-1"، torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
negative_prompt = "low quality, bad quality"

image = pipeline(prompt=prompt، negative_prompt=negative_prompt، prior_guidance_scale=1.0، guidance_scale=4.0، height=768، width=768).images[0]
image
```

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder"، torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
negative_prompt = "low quality, bad quality"

image = pipeline(prompt=prompt، negative_prompt=negative_prompt، prior_guidance_scale=1.0، guidance_scale=4.0، height=768، width=768).images[0]
image
```

</hfoption>

</hfoptions>
بالتأكيد! فيما يلي ترجمة للنص الموجود في الفقرات والعناوين مع الحفاظ على تنسيق Markdown:

## Image-to-image

لتحويل الصورة إلى صورة، قم بتمرير الصورة الأولية وملاحظة النص لتحديد الصورة إلى خط الأنابيب. ابدأ بتحميل خط أنابيب سابق:

لتحميل صورة للشرط عليها:

```py
from diffusers.utils import load_image

# قم بتنزيل الصورة
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
original_image = load_image(url)
original_image = original_image.resize((768, 512))
```

![الصورة الأصلية](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg)

قم بتوليد `image_embeds` و `negative_image_embeds` مع خط أنابيب سابق:

```py
prompt = "منظر طبيعي خيالي، إضاءة سينمائية"
negative_prompt = "جودة منخفضة، جودة سيئة"

image_embeds, negative_image_embeds = prior_pipeline(prompt, negative_prompt).to_tuple()
```

الآن قم بتمرير الصورة الأصلية، وجميع المطالبات والترميز إلى خط الأنابيب لإنشاء صورة:

لاستخدام [`AutoPipelineForImage2Image`] للاتصال تلقائيًا بخطوط الأنابيب المجمعة:

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True)
pipeline.enable_model_cpu_offload()

prompt = "منظر طبيعي خيالي، إضاءة سينمائية"
negative_prompt = "جودة منخفضة، جودة سيئة"

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
original_image = load_image(url)

original_image.thumbnail((768, 768))

image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=original_image, strength=0.3).images[0]
make_image_grid([original_image.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
```

![نتيجة الصورة إلى صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-docs/img2img_fantasyland.png)

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

prompt = "منظر طبيعي خيالي، إضاءة سينمائية"
negative_prompt = "جودة منخفضة، جودة سيئة"

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
original_image = load_image(url)

original_image.thumbnail((768, 768))

image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=original_image, strength=0.3).images[0]
make_image_grid([original_image.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
```

![نتيجة الصورة إلى صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-image-to-image.png)
## إكمال الصور

للقيام بإكمال الصور، ستحتاج إلى الصورة الأصلية، وقناع للمنطقة المراد استبدالها في الصورة الأصلية، ونص يوضح ما تريد إكماله. قم بتحميل خط الأنابيب الأولي:

```py
from diffusers import KandinskyInpaintPipeline, KandinskyPriorPipeline
from diffusers.utils import load_image, make_image_grid
import torch
import numpy as np
from PIL import Image

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = KandinskyInpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
```

قم بتحميل صورة أولية وإنشاء قناع:

```py
init_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
mask = np.zeros((768, 768), dtype=np.float32)
# قناع المنطقة أعلى رأس القطة
mask[:250, 250:-250] = 1
```

قم بتوليد التضمين باستخدام خط أنابيب الأولي:

```py
prompt = "قبعة"
prior_output = prior_pipeline(prompt)
```

الآن قم بتمرير الصورة الأولية والقناع والنص والتضمين إلى خط الأنابيب لإنشاء صورة:

```py
output_image = pipeline(prompt, image=init_image, mask_image=mask, **prior_output, height=768, width=768, num_inference_steps=150).images[0]
mask = Image.fromarray((mask*255).astype('uint8'), 'L')
make_image_grid([init_image, mask, output_image], rows=1, cols=3)
```

يمكنك أيضًا استخدام خط أنابيب [`KandinskyInpaintCombinedPipeline`] الشامل من البداية إلى النهاية و [`KandinskyV22InpaintCombinedPipeline`] لاستدعاء خطي الأنابيب الأولي وفك التشفير معًا في الخلفية. استخدم [`AutoPipelineForInpainting`] لهذا:

```py
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipe = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

init_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
mask = np.zeros((768, 768), dtype=np.float32)
# قناع المنطقة أعلى رأس القطة
mask[:250, 250:-250] = 1
prompt = "قبعة"

output_image = pipe(prompt=prompt, image=init_image, mask_image=mask).images[0]
mask = Image.fromarray((mask*255).astype('uint8'), 'L')
make_image_grid([init_image, mask, output_image], rows=1, cols=3)
```

## الاستيفاء

يتيح الاستيفاء استكشاف الفراغات بين تضمين الصورة والنص، وهي طريقة رائعة لمشاهدة بعض المخرجات الوسيطة لنماذج الأولية. قم بتحميل خط أنابيب الأولية وصورتين تريد الاستيفاء بينهما:

```py
from diffusers import KandinskyPriorPipeline, KandinskyPipeline
from diffusers.utils import load_image, make_image_grid
import torch

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
img_1 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
img_2 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg")
make_image_grid([img_1.resize((512,512)), img_2.resize((512,512))], rows=1, cols=2)
```

حدد النصوص أو الصور التي تريد الاستيفاء بينها، وقم بتعيين الأوزان لكل نص أو صورة. جرب مع الأوزان المختلفة لمعرفة كيفية تأثيرها على الاستيفاء!

```py
images_texts = ["قطة", img_1, img_2]
weights = [0.3, 0.3, 0.4]
```

قم بالاتصال بوظيفة `interpolate` لتوليد التضمينات، ثم قم بتمريرها إلى خط الأنابيب لإنشاء الصورة:

```py
# يمكن ترك النص فارغًا
prompt = ""
prior_out = prior_pipeline.interpolate(images_texts, weights)

pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

image = pipeline(prompt, **prior_out, height=768, width=768).images[0]
image
```
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع مراعاة التعليمات التي قدمتها.

## ControlNet

تمكّن ControlNet من تهيئة النماذج المُدربة مسبقًا الكبيرة باستخدام مدخلات إضافية مثل خريطة العمق أو كشف الحواف. على سبيل المثال، يمكنك تهيئة Kandinsky 2.2 باستخدام خريطة عمق حتى يفهم النموذج ويحافظ على بنية صورة العمق.

دعونا نقوم بتحميل صورة واستخراج خريطة العمق الخاصة بها:

الصورة:

ثم يمكنك استخدام "التقدير العميقة" من Transformers من Hugging Face لمعالجة الصورة واسترداد خريطة العمق:

### Text-to-image

قم بتحميل خط الأنابيب المسبق والخط KandinskyV22ControlnetPipeline:

قم بتوليد تضمين الصورة من موجه ومحث سلبي:

أخيرًا، قم بتمرير تضمين الصورة وصورة العمق إلى KandinskyV22ControlnetPipeline لتوليد صورة:

الصورة:

### Image-to-image

بالنسبة للصورة إلى الصورة باستخدام ControlNet، ستحتاج إلى استخدام ما يلي:

- KandinskyV22PriorEmb2EmbPipeline لتوليد تضمين الصورة من موجه نصي وصورة أولية
- KandinskyV22ControlnetImg2ImgPipeline لتوليد صورة من الصورة الأولية وتضمين الصورة

قم بمعالجة واستخراج خريطة عمق لصورة أولية لقطة باستخدام "التقدير العميقة" من Transformers من Hugging Face:

قم بتحميل خط الأنابيب المسبق وKandinskyV22ControlnetImg2ImgPipeline:

قم بتمرير موجه نصي وصورة أولية إلى خط الأنابيب المسبق لتوليد تضمين الصورة:

الآن يمكنك تشغيل KandinskyV22ControlnetImg2ImgPipeline لتوليد صورة من الصورة الأولية وتضمين الصورة:

الصورة:

## التحسينات

Kandinsky فريد من نوعه لأنه يتطلب خط أنابيب مسبق لتوليد الخرائط، وخط أنابيب ثانٍ لترميز المحفزات إلى صورة. يجب تركيز جهود التحسين على خط الأنابيب الثاني لأنه يتم فيه الجزء الأكبر من الحساب. فيما يلي بعض النصائح لتحسين Kandinsky أثناء الاستدلال.

1. قم بتمكين xFormers إذا كنت تستخدم PyTorch < 2.0:

2. قم بتمكين "التورط" إذا كنت تستخدم PyTorch >= 2.0 لاستخدام الانتباه إلى نقطة المنتج المقياس (SDPA) تلقائيًا:

هذا هو نفسه عند تعيين معالج الاهتمام صراحة لاستخدام ~models.attention_processor.AttnAddedKVProcessor2_0:

قم بإلغاء تحميل النموذج إلى وحدة المعالجة المركزية باستخدام ~KandinskyPriorPipeline.enable_model_cpu_offload لتجنب أخطاء نفاد الذاكرة:

4. يستخدم خط أنابيب النص إلى الصورة، بشكل افتراضي، DDIMScheduler، ولكن يمكنك استبداله بمخطط آخر مثل DDPMScheduler لمعرفة كيفية تأثير ذلك على المقايضة بين سرعة الاستدلال وجودة الصورة: