# كاندينسكي

[[open-in-colab]]

تعد نماذج كاندينسكي سلسلة من نماذج توليد الصور متعددة اللغات. يستخدم نموذج كاندينسكي 2.0 مشفرين نصيين متعددين اللغات ويقوم بدمج تلك النتائج لـ UNet.

يغير [كاندينسكي 2.1] (../ api/pipelines/kandinsky) البنية لتضمين نموذج صورة أولية ([` CLIP `](https://huggingface.co/docs/transformers/model_doc/clip)) لتوليد خريطة بين النصوص وتضمين الصور. توفر الخريطة محاذاة أفضل للنص والصورة ويتم استخدامها مع تضمين النص أثناء التدريب، مما يؤدي إلى نتائج عالية الجودة. وأخيرًا، يستخدم كاندينسكي 2.1 [مشفرًا لـ Modulating Quantized Vectors (MoVQ)](https://huggingface.co/papers/2209.09002) - والذي يضيف طبقة تطبيع شرطي مكاني لزيادة الواقعية الفوتوغرافية - لترميز الكمونات إلى صور.

يحسن [كاندينسكي 2.2] (../ api/pipelines/kandinsky_v22) النموذج السابق عن طريق استبدال مشفر الصورة في نموذج الصورة الأولية بنموذج CLIP-ViT-G أكبر لتحسين الجودة. كما أعيد تدريب نموذج الصورة الأولية على صور بدقة وجوانب مختلفة لتوليد صور عالية الدقة وأحجام صور مختلفة.

يبسط [كاندينسكي 3] (../ api/pipelines/kandinsky3) البنية وينتقل بعيدًا عن عملية التوليد ذات المرحلتين التي تتضمن النموذج الأولي ونموذج الانتشار. بدلاً من ذلك، يستخدم Kandinsky 3 [Flan-UL2](https://huggingface.co/google/flan-ul2) لتشفير النص، و UNet مع [BigGan-deep](https://hf.co/papers/1809.11096) blocks، و [Sber-MoVQGAN](https://github.com/ai-forever/MoVQGAN) لترميز الكمونات إلى صور. يتم تحقيق فهم النص وجودة الصورة المولدة بشكل أساسي من خلال استخدام مشفر نصي وشبكة UNet أكبر.

سيوضح لك هذا الدليل كيفية استخدام نماذج كاندينسكي للمهام مثل النص إلى الصورة والصورة إلى الصورة، والطلاء، والتنقل، والمزيد.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers transformers accelerate
```

<Tip warning={true}>

يعد استخدام Kandinsky 2.1 و 2.2 متشابهًا جدًا! الفرق الوحيد هو أن Kandinsky 2.2 لا يقبل `prompt` كإدخال عند فك تشفير الكمونات. بدلاً من ذلك، يقبل Kandinsky 2.2 فقط `image_embeds` أثناء فك التشفير.

<br>

لدى Kandinsky 3 بنية أكثر إحكاما ولا يتطلب نموذجًا أوليًا. وهذا يعني أن استخدامه مطابق لنماذج الانتشار الأخرى مثل [Stable Diffusion XL] (sdxl).

</Tip>

## نص إلى صورة

لاستخدام نماذج كاندينسكي لأي مهمة، ابدأ دائمًا بإعداد خط أنابيب النموذج الأولي لتشفير الفكرة وتوليد تضمين الصورة. كما يقوم خط أنابيب النموذج الأولي بتوليد `negative_image_embeds` التي تتوافق مع الفكرة السلبية `""`. للحصول على نتائج أفضل، يمكنك تمرير `negative_prompt` الفعلي إلى خط أنابيب النموذج الأولي، ولكن هذا سيزيد من حجم دفعة النموذج الأولي الفعالة بمقدار 2x.

<hfoptions id="text-to-image">
<hfoption id="Kandinsky 2.1">

```py
from diffusers import KandinskyPriorPipeline, KandinskyPipeline
import torch

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16).to("cuda")
pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1"، torch_dtype=torch.float16).to("cuda")

الفكرة = "مخلوق برجر الجبن الغريب الذي يأكل نفسه، claymation، السينمائي، الإضاءة المزاجية"
negative_prompt = "جودة منخفضة، جودة سيئة" # اختياري لتضمين فكرة سلبية، ولكن النتائج تكون أفضل عادة
image_embeds، negative_image_embeds = prior_pipeline(prompt، negative_prompt، guidance_scale=1.0).to_tuple()
```

الآن قم بتمرير جميع الأفكار والتضمينات إلى [`KandinskyPipeline`] لتوليد صورة:

```py
الصورة = pipeline(prompt، image_embeds=image_embeds، negative_prompt=negative_prompt، negative_image_embeds=negative_image_embeds، height=768، width=768).images[0]
الصورة
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

الفكرة = "مخلوق برجر الجبن الغريب الذي يأكل نفسه، claymation، السينمائي، الإضاءة المزاجية"
negative_prompt = "جودة منخفضة، جودة سيئة" # اختياري لتضمين فكرة سلبية، ولكن النتائج تكون أفضل عادة
image_embeds، negative_image_embeds = prior_pipeline(prompt، guidance_scale=1.0).to_tuple()
```

مرر `image_embeds` و` negative_image_embeds` إلى [`KandinskyV22Pipeline`] لتوليد صورة:

```py
الصورة = pipeline(image_embeds=image_embeds، negative_image_embeds=negative_image_embeds، height=768، width=768).images[0]
الصورة
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-text-to-image.png"/>
</div>

</hfoption>
<hfoption id="Kandinsky 3">

لا يتطلب Kandinsky 3 نموذجًا أوليًا، لذا يمكنك تحميل [`Kandinsky3Pipeline`] مباشرة وتمرير فكرة لتوليد صورة:

```py
from diffusers import Kandinsky3Pipeline
import torch

pipeline = Kandinsky3Pipeline.from_pretrained("kandinsky-community/kandinsky-3"، variant="fp16"، torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

الفكرة = "مخلوق برجر الجبن الغريب الذي يأكل نفسه، claymation، السينمائي، الإضاءة المزاجية"
الصورة = pipeline(prompt).images[0]
الصورة
```

</hfoption>
</hfoptions>

يوفر 🤗 Diffusers أيضًا واجهة برمجة تطبيقات (API) شاملة مع [`KandinskyCombinedPipeline`] و [`KandinskyV22CombinedPipeline`]، مما يعني أنه لا يلزم تحميل خط أنابيب النموذج الأولي وخط أنابيب النص إلى الصورة بشكل منفصل. يقوم خط الأنابيب المجمع بتحميل كل من النموذج الأولي وفك التشفير تلقائيًا. لا يزال بإمكانك تعيين قيم مختلفة لخط أنابيب النموذج الأولي باستخدام معلمات `prior_guidance_scale` و` prior_num_inference_steps` إذا كنت تريد ذلك.

استخدم [`AutoPipelineForText2Image`] لاستدعاء خطوط الأنابيب المجمعة تلقائيًا:

<hfoptions id="text-to-image">
<hfoption id="Kandinsky 2.1">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-1"، torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

الفكرة = "مخلوق برجر الجبن الغريب الذي يأكل نفسه، claymation، السينمائي، الإضاءة المزاجية"
negative_prompt = "جودة منخفضة، جودة سيئة"

الصورة = pipeline(prompt=prompt، negative_prompt=negative_prompt، prior_guidance_scale=1.0، guidance_scale=4.0، height=768، width=768).images[0]
الصورة
```

</hfoption>
<hfoption id="Kandinsky 2.2">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder"، torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

الفكرة = "مخلوق برجر الجبن الغريب الذي يأكل نفسه، claymation، السينمائي، الإضاءة المزاجية"
negative_prompt = "جودة منخفضة، جودة سيئة"

الصورة = pipeline(prompt=prompt، negative_prompt=negative_prompt، prior_guidance_scale=1.0، guidance_scale=4.0، height=768، width=768).images[0]
الصورة
```

</hfoption>
</hfoptions>
لم يتم ترجمة الأجزاء التي تحتوي على أكواد برمجية ورموز HTML وCSS بناء على طلبك.

## Image-to-image

بالنسبة للصورة إلى الصورة، قم بتمرير الصورة الأولية وطلب النص لتهيئة الصورة إلى خط الأنابيب. ابدأ بتحميل خط الأنابيب السابق:

<hfoptions id="image-to-image">

<hfoption id="Kandinsky 2.1">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

<hfoption id="Kandinsky 3">

لا يحتاج Kandinsky 3 إلى نموذج أولي، لذلك يمكنك تحميل خط أنابيب الصورة إلى الصورة مباشرة:

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

</hfoptions>

قم بتنزيل صورة للتهيئة:

```py
# لم يتم ترجمة الأكواد البرمجية
```

![صورة لجبال مرسومة بالخطوط العريضة](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg)

قم بتوليد `image_embeds` و `negative_image_embeds` مع خط أنابيب الأولية:

```py
طلب النص = "منظر خيالي، إضاءة سينمائية"
negative_prompt = "جودة منخفضة، جودة سيئة"

image_embeds, negative_image_embeds = prior_pipeline(prompt, negative_prompt).to_tuple()
```

الآن قم بتمرير الصورة الأصلية، وجميع المطالبات والترميز إلى خط الأنابيب لإنشاء صورة:

<hfoptions id="image-to-image">

<hfoption id="Kandinsky 2.1">

```py
# لم يتم ترجمة الأكواد البرمجية
```

![صورة لمنظر خيالي](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-docs/img2img_fantasyland.png)

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
# لم يتم ترجمة الأكواد البرمجية
```

![صورة لمنظر طبيعي باستخدام Kandinsky 2.2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-image-to-image.png)

</hfoption>

<hfoption id="Kandinsky 3">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

</hfoptions>

يوفر 🤗 Diffusers أيضًا واجهة برمجة تطبيقات (API) شاملة مع [`KandinskyImg2ImgCombinedPipeline`] و [`KandinskyV22Img2ImgCombinedPipeline`]، مما يعني أنه لا يلزم تحميل خط أنابيب الأولية وخط أنابيب الصورة إلى الصورة بشكل منفصل. يقوم خط الأنابيب المشترك تلقائيًا بتحميل كل من النموذج الأولي والمحلل. لا يزال بإمكانك تعيين قيم مختلفة لخط أنابيب الأولية باستخدام معلمات `prior_guidance_scale` و `prior_num_inference_steps` إذا كنت تريد ذلك.

استخدم [`AutoPipelineForImage2Image`] لاستدعاء خطوط الأنابيب المشتركة تلقائيًا:

<hfoptions id="image-to-image">

<hfoption id="Kandinsky 2.1">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

</hfoptions>

## Inpainting

<Tip warning={true}>

⚠️ تستخدم نماذج Kandinsky الآن ⬜️ **pixels البيضاء** لتمثيل المنطقة المطلية بدلاً من البكسلات السوداء. إذا كنت تستخدم [`KandinskyInpaintPipeline`] في الإنتاج، فيجب عليك تغيير القناع لاستخدام البكسلات البيضاء:

```py
# لم يتم ترجمة الأكواد البرمجية
```

</Tip>

بالنسبة للرسم، ستحتاج إلى الصورة الأصلية، وقناع للمنطقة التي سيتم استبدالها في الصورة الأصلية، وطلب نص لما سيتم طلاؤه. قم بتحميل خط أنابيب الأولية:

<hfoptions id="inpaint">

<hfoption id="Kandinsky 2.1">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

</hfoptions>

قم بتحميل صورة أولية وإنشاء قناع:

```py
# لم يتم ترجمة الأكواد البرمجية
```

قم بتوليد الترميز باستخدام خط أنابيب الأولية:

```py
طلب النص = "قبعة"
prior_output = prior_pipeline(prompt)
```

الآن قم بتمرير الصورة الأولية والقناع والطلب والترميز إلى خط الأنابيب لإنشاء صورة:

<hfoptions id="inpaint">

<hfoption id="Kandinsky 2.1">

```py
# لم يتم ترجمة الأكواد البرمجية
```

![صورة لقطة بقبعة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-docs/inpaint_cat_hat.png)

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
# لم يتم ترجمة الأكواد البرمجية
```

![صورة لقطة بقبعة باستخدام Kandinsky 2.2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinskyv22-inpaint.png)

</hfoption>

</hfoptions>

يمكنك أيضًا استخدام [`KandinskyInpaintCombinedPipeline`] الشامل و [`KandinskyV22InpaintCombinedPipeline`] لاستدعاء خطوط أنابيب الأولية والمحلل معًا تحت غطاء المحرك. استخدم [`AutoPipelineForInpainting`] لهذا الغرض:

<hfoptions id="inpaint">

<hfoption id="Kandinsky 2.1">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

<hfoption id="Kandinsky 2.2">

```py
# لم يتم ترجمة الأكواد البرمجية
```

</hfoption>

</hfoptions>
## الاستيفاء

يتيح الاستيفاء استكشاف الفراغ الكامن بين تضمين الصور والنصوص، وهي طريقة رائعة لمشاهدة بعض المخرجات الوسيطة للنموذج السابق. قم بتحميل خط الأنابيب السابق وصورتين تريد الاستيفاء بينهما:

```py
from diffusers import KandinskyPriorPipeline, KandinskyPipeline
from diffusers.utils import load_image, make_image_grid
import torch

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
img_1 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
img_2 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg")
make_image_grid([img_1.resize((512,512)), img_2.resize((512,512))], rows=1, cols=2)
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"/>
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg"/>
</div>

حدد النصوص أو الصور التي تريد الاستيفاء بينها، وقم بضبط الأوزان لكل نص أو صورة. جرب مع الأوزان المختلفة لترى كيف تؤثر على الاستيفاء!

```py
images_texts = ["a cat", img_1, img_2]
weights = [0.3, 0.3, 0.4]
```

قم بالاستدعاء على دالة `interpolate` لتوليد التضمينات، ثم مررها إلى خط الأنابيب لتوليد الصورة:

```py
# يمكن ترك المحث فارغًا
prompt = ""
prior_out = prior_pipeline.interpolate(images_texts, weights)

pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

image = pipeline(prompt, **prior_out, height=768, width=768).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/kandinsky-docs/starry_cat.png"/>
</div>

## ControlNet

<Tip warning={true}>
⚠️ ControlNet مدعوم فقط لـ Kandinsky 2.2!
</Tip>

تمكّن ControlNet من تهيئة النماذج الكبيرة مسبقة التدريب مع مدخلات إضافية مثل خريطة العمق أو كشف الحواف. على سبيل المثال، يمكنك تهيئة Kandinsky 2.2 باستخدام خريطة العمق بحيث يفهم النموذج ويحافظ على بنية صورة العمق.

قم بتحميل صورة واستخراج خريطة العمق الخاصة بها:

```py
from diffusers.utils import load_image

img = load_image(
"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"
).resize((768, 768))
img
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"/>
</div>

بعد ذلك، يمكنك استخدام خط أنابيب `depth-estimation` من 🤗 Transformers لمعالجة الصورة واسترداد خريطة العمق:

```py
import torch
import numpy as np

from transformers import pipeline

def make_hint(image, depth_estimator):
image = depth_estimator(image)["depth"]
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
detected_map = torch.from_numpy(image).float() / 255.0
hint = detected_map.permute(2, 0, 1)
return hint

depth_estimator = pipeline("depth-estimation")
hint = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")
```

### من نص إلى صورة [[controlnet-text-to-image]]

قم بتحميل خط أنابيب سابق وخط أنابيب [`KandinskyV22ControlnetPipeline`]:

```py
from diffusers import KandinskyV22PriorPipeline, KandinskyV22ControlnetPipeline

prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
"kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipeline = KandinskyV22ControlnetPipeline.from_pretrained(
"kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
).to("cuda")
```

قم بتوليد تضمينات الصورة من محث ومحث سابق:

```py
prompt = "A robot, 4k photo"
negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

generator = torch.Generator(device="cuda").manual_seed(43)

image_emb, zero_image_emb = prior_pipeline(
prompt=prompt, negative_prompt=negative_prior_prompt, generator=generator
).to_tuple()
```

أخيرًا، مرر تضمينات الصورة وصورة العمق إلى خط أنابيب [`KandinskyV22ControlnetPipeline`] لتوليد صورة:

```py
image = pipeline(image_embeds=image_emb, negative_image_embeds=zero_image_emb, hint=hint, num_inference_steps=50, generator=generator, height=768, width=768).images[0]
image
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/robot_cat_text2img.png"/>
</div>

### من صورة إلى صورة [[controlnet-image-to-image]]

بالنسبة للتحويل من صورة إلى صورة باستخدام ControlNet، ستحتاج إلى استخدام:

- [`KandinskyV22PriorEmb2EmbPipeline`] لتوليد تضمينات الصورة من محث نصي وصورة
- [`KandinskyV22ControlnetImg2ImgPipeline`] لتوليد صورة من الصورة الأولية وتضمينات الصورة

قم بمعالجة صورة أولية لقطة مستخدماً خط أنابيب `depth-estimation` من 🤗 Transformers واستخرج خريطة العمق:

```py
import torch
import numpy as np

from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
from diffusers.utils import load_image
from transformers import pipeline

img = load_image(
"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"
).resize((768, 768))

def make_hint(image, depth_estimator):
image = depth_estimator(image)["depth"]
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
detected_map = torch.from_numpy(image).float() / 255.0
hint = detected_map.permute(2, 0, 1)
return hint

depth_estimator = pipeline("depth-estimation")
hint = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")
```

قم بتحميل خط الأنابيب السابق وخط أنابيب [`KandinskyV22ControlnetImg2ImgPipeline`]:

```py
prior_pipeline = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
"kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipeline = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
"kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
).to("cuda")
```

مرر محث نصي وصورة أولية إلى خط الأنابيب السابق لتوليد تضمينات الصورة:

```py
prompt = "A robot, 4k photo"
negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

generator = torch.Generator(device="cuda").manual_seed(43)

img_emb = prior_pipeline(prompt=prompt, image=img, strength=0.85, generator=generator)
negative_emb = prior_pipeline(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)
```

الآن يمكنك تشغيل خط أنابيب [`KandinskyV22ControlnetImg2ImgPipeline`] لتوليد صورة من الصورة الأولية وتضمينات الصورة:

```py
image = pipeline(image=img, strength=0.5, image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb.image_embeds, hint=hint, num_inference_steps=50, generator=generator, height=768, width=768).images[0]
make_image_grid([img.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
```

<div class="flex justify-center">
<img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/robot_cat.png"/>
</div>

## التحسينات

يتميز Kandinsky بكونه فريداً من نوعه لأنه يتطلب خط أنابيب سابق لتوليد التضمينات، وخط أنابيب ثانٍ لفك تشفير الكمونات إلى صورة. يجب أن تركز جهود التحسين على خط الأنابيب الثاني لأن هذا هو المكان الذي يتم فيه الجزء الأكبر من الحساب. فيما يلي بعض النصائح لتحسين Kandinsky أثناء الاستدلال.

1. قم بتمكين [xFormers](../optimization/xformers) إذا كنت تستخدم PyTorch < 2.0:

```diff
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
+ pipe.enable_xformers_memory_efficient_attention()
```

2. قم بتمكين `torch.compile` إذا كنت تستخدم PyTorch >= 2.0 لاستخدام الانتباه إلى النقطة المحددة المقياس (SDPA) تلقائيًا:

```diff
pipe.unet.to(memory_format=torch.channels_last)
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

وهذا مماثل لتحديد معالج الانتباه لاستخدام [`~models.attention_processor.AttnAddedKVProcessor2_0`] بشكل صريح:

```py
from diffusers.models.attention_processor import AttnAddedKVProcessor2_0

pipe.unet.set_attn_processor(AttnAddedKVProcessor2_0())
```

3. قم بتفريغ النموذج إلى وحدة المعالجة المركزية باستخدام [`~KandinskyPriorPipeline.enable_model_cpu_offload`] لتجنب أخطاء نفاد الذاكرة:

```diff
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
+ pipe.enable_model_cpu_offload()
```

4. يستخدم خط أنابيب من نص إلى صورة، بشكل افتراضي، جدول DDIMScheduler، ولكن يمكنك استبداله بجدول آخر مثل [`DDPMScheduler`] لمعرفة كيفية تأثير ذلك على المقايضة بين سرعة الاستدلال وجودة الصورة:

```py
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline

scheduler = DDPMScheduler.from_pretrained("kandinsky-community/kandinsky-2-1", subfolder="ddpm_scheduler")
pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
```