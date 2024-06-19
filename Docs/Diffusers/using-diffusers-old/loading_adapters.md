# تحميل المحولات 

هناك العديد من تقنيات [التدريب](../training/overview) لتخصيص نماذج الانتشار لتوليد صور لموضوع محدد أو صور بأساليب معينة. تنتج كل طريقة تدريب من هذه الطرق نوعًا مختلفًا من المحول. يقوم بعض المحولات بتوليد نموذج جديد تمامًا، بينما يقوم البعض الآخر بتعديل مجموعة فرعية فقط من التعليقات التوضيحية أو الأوزان. وهذا يعني أن عملية التحميل لكل محول مختلفة أيضًا.

سيوضح هذا الدليل لك كيفية تحميل أوزان DreamBooth وinversion النصية وLoRA.

## DreamBooth

[DreamBooth](https://dreambooth.github.io/) يضبط دقيقًا *نموذج انتشار كامل* على مجرد عدة صور لموضوع لتوليد صور لهذا الموضوع بأساليب وإعدادات جديدة. تعمل هذه الطريقة من خلال استخدام كلمة خاصة في المطالبة التي يتعلمها النموذج لربطها بصورة الموضوع. من بين جميع طرق التدريب، ينتج DreamBooth أكبر حجم ملف (عادةً بضع جيجابايت) لأنه نموذج نقطة تفتيش كامل.

دعونا نحمل نقطة تفتيش [herge_style](https://huggingface.co/sd-dreambooth-library/herge-style)، والتي يتم تدريبها على 10 صور فقط رسمها Hergé، لتوليد الصور بتلك الطريقة. لكي يعمل، تحتاج إلى تضمين الكلمة الخاصة `herge_style` في مطالبتك لتشغيل نقطة التفتيش:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("sd-dreambooth-library/herge-style", torch_dtype=torch.float16).to("cuda")
prompt = "A cute herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

## الانقلاب النصي

[الانقلاب النصي](https://textual-inversion.github.io/) يشبه إلى حد كبير DreamBooth ويمكنه أيضًا تخصيص نموذج الانتشار لتوليد مفاهيم معينة (الأنماط، الأشياء) من مجرد بضع صور. تعمل هذه الطريقة من خلال تدريب وإيجاد تعليقات توضيحية جديدة تمثل الصور التي تقدمها مع كلمة خاصة في المطالبة. ونتيجة لذلك، تظل أوزان نموذج الانتشار كما هي وتنتج عملية التدريب ملفًا صغيرًا نسبيًا (بضع كيلوبايتات).

نظرًا لأن الانقلاب النصي يقوم بإنشاء تعليقات توضيحية، فإنه لا يمكن استخدامه بمفرده مثل DreamBooth ويتطلب نموذجًا آخر.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

الآن يمكنك تحميل التعليقات التوضيحية للانقلاب النصي باستخدام طريقة [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] وتوليد بعض الصور. دعونا نحمل التعليقات التوضيحية [sd-concepts-library/gta5-artwork](https://huggingface.co/sd-concepts-library/gta5-artwork) وتحتاج إلى تضمين الكلمة الخاصة `<gta5-artwork>` في مطالبتك لتشغيلها:

```py
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, <gta5-artwork> style"
image = pipeline(prompt).images[0]
image
```

يمكن أيضًا تدريب الانقلاب النصي على أشياء غير مرغوب فيها لإنشاء تعليقات توضيحية *سلبية* لمنع النموذج من إنشاء صور بتلك الأشياء غير المرغوب فيها مثل الصور الضبابية أو الأصابع الإضافية على اليد. يمكن أن يكون هذا طريقة سهلة لتحسين مطالبتك بسرعة. ستقوم أيضًا بتحميل التعليقات التوضيحية باستخدام [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]، ولكن هذه المرة، ستحتاج إلى معلمتين أخريين:

- `weight_name`: يحدد اسم ملف الوزن لتحميله إذا تم حفظ الملف بتنسيق 🤗 Diffusers باسم محدد أو إذا تم تخزين الملف بتنسيق A1111
- `token`: يحدد الكلمة الخاصة التي سيتم استخدامها في المطالبة لتشغيل التعليقات التوضيحية

دعونا نحمل التعليقات التوضيحية [sayakpaul/EasyNegative-test](https://huggingface.co/sayakpaul/EasyNegative-test):

```py
pipeline.load_textual_inversion(
"sayakpaul/EasyNegative-test", weight_name="EasyNegative.safetensors", token="EasyNegative"
)
```

الآن يمكنك استخدام `token` لتوليد صورة بالتعليقات التوضيحية السلبية:

```py
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, EasyNegative"
negative_prompt = "EasyNegative"

image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
image
```

## لورا

[التكيف منخفض الرتبة (LoRA)](https://huggingface.co/papers/2106.09685) هي تقنية تدريب شائعة لأنها سريعة وتولد أحجام ملفات أصغر (بضع مئات الميغابايت). مثل الطرق الأخرى في هذا الدليل، يمكن لـ LoRA تدريب نموذج على تعلم أنماط جديدة من مجرد بضع صور. تعمل عن طريق إدراج أوزان جديدة في نموذج الانتشار ثم تدريب الأوزان الجديدة فقط بدلاً من النموذج بأكمله. يجعل هذا الأمر أسرع في التدريب وأسهل في التخزين.

> ملاحظة: LoRA هي تقنية تدريب عامة جدًا يمكن استخدامها مع طرق التدريب الأخرى. على سبيل المثال، من الشائع تدريب نموذج باستخدام DreamBooth وLoRA. كما أنه من الشائع بشكل متزايد تحميل ودمج العديد من LoRAs لإنشاء صور جديدة وفريدة من نوعها. يمكنك معرفة المزيد عنها في دليل [دمج LoRAs](merge_loras) المتعمق نظرًا لأن الدمج يقع خارج نطاق دليل التحميل هذا.

تحتاج LoRAs أيضًا إلى استخدامها مع نموذج آخر:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
```

ثم استخدم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora) وحدد اسم ملف الأوزان من المستودع:

```py
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors")
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]
image
```

تقوم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] بتحميل أوزان LoRA في كل من UNet ومشفر النص. إنها الطريقة المفضلة لتحميل LoRAs لأنها يمكن أن تتعامل مع الحالات التي:

- لا تحتوي أوزان LoRA على محددات منفصلة لـ UNet ومشفر النص
- تحتوي أوزان LoRA على محددات منفصلة لـ UNet ومشفر النص

ولكن إذا كنت بحاجة فقط إلى تحميل أوزان LoRA في UNet، فيمكنك استخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]. دعونا نحمل LoRA [jbilcke-hf/sdxl-cinematic-1](https://huggingface.co/jbilcke-hf/sdxl-cinematic-1):

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")

# use cnmt في المطالبة لتشغيل LoRA
prompt = "A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

لإلغاء تحميل أوزان LoRA، استخدم طريقة [`~loaders.LoraLoaderMixin.unload_lora_weights`] للتخلص من أوزان LoRA واستعادة النموذج إلى أوزانه الأصلية:

```py
pipeline.unload_lora_weights()
```
### ضبط مقياس وزن LoRA

بالنسبة لكل من [`~loaders.LoraLoaderMixin.load_lora_weights`] و [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]، يمكنك تمرير معلمة `cross_attention_kwargs={"scale": 0.5}` لضبط مقدار أوزان LoRA التي تريد استخدامها. وتعتبر قيمة `0` مماثلة لاستخدام أوزان نموذج الأساس فقط، بينما تعادل قيمة `1` استخدام LoRA الدقيق الضبط بالكامل.

وللتحكم بشكل أكثر دقة في مقدار أوزان LoRA المستخدمة في كل طبقة، يمكنك استخدام [`~loaders.LoraLoaderMixin.set_adapters`] وتمرير قاموس يحدد مقدار المقياس المستخدم في أوزان كل طبقة.

```python
pipe = ... # إنشاء خط الأنابيب
pipe.load_lora_weights(..., adapter_name="my_adapter")
scales = {
"text_encoder": 0.5,
"text_encoder_2": 0.5,  # يمكن استخدامه فقط إذا كان لدى pipe مشفر نص ثانوي
"unet": {
"down": 0.9,  # ستستخدم جميع المحولات في الجزء السفلي المقياس 0.9
# "mid"  # في هذا المثال، لم يتم إعطاء "mid"، لذلك ستستخدم جميع المحولات في الجزء الأوسط المقياس الافتراضي 1.0
"up": {
"block_0": 0.6,  # ستستخدم جميع المحولات الثلاثة في الكتلة 0 في الجزء العلوي المقياس 0.6
"block_1": [0.4, 0.8, 1.0]، # ستستخدم المحولات الثلاثة في الكتلة 1 في الجزء العلوي المقاييس 0.4 و 0.8 و 1.0 على التوالي
}
}
}
pipe.set_adapters("my_adapter", scales)
```

ويعمل هذا أيضًا مع عدة محولات - راجع [هذا الدليل](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#customize-adapters-strength) لمعرفة كيفية القيام بذلك.

<Tip warning={true}>
حاليًا، يدعم [`~loaders.LoraLoaderMixin.set_adapters`] فقط مقاييس أوزان الاهتمام. إذا كان لدى LoRA أجزاء أخرى (مثل شبكات ResNet أو down-/upsamplers)، فستحتفظ بمقياس 1.0.
</Tip>

### Kohya وTheLastBen

تشمل برامج تدريب LoRA الشهيرة الأخرى من المجتمع تلك التي قام بها [Kohya](https://github.com/kohya-ss/sd-scripts/) و[TheLastBen](https://github.com/TheLastBen/fast-stable-diffusion). وتنشئ برامج التدريب هذه نقاط تفتيش LoRA مختلفة عن تلك التي تدربها 🤗 Diffusers، ولكن يمكن تحميلها بنفس الطريقة.

<hfoptions id="other-trainers">
<hfoption id="Kohya">

لتحميل نقطة تفتيش LoRA من Kohya، دعنا نقوم بتنزيل نقطة تفتيش [Blueprintify SD XL 1.0](https://civitai.com/models/150986/blueprintify-sd-xl-10) كمثال من [Civitai](https://civitai.com/):

```sh
!wget https://civitai.com/api/download/models/168776 -O blueprintify-sd-xl-10.safetensors
```

قم بتحميل نقطة تفتيش LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]، وحدد اسم الملف في معلمة `weight_name`:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/weights", weight_name="blueprintify-sd-xl-10.safetensors")
```

قم بتوليد صورة:

```py
# استخدم bl3uprint في المطالبة لتشغيل LoRA
prompt = "bl3uprint, a highly detailed blueprint of the eiffel tower, explaining how to build all parts, many txt, blueprint grid backdrop"
image = pipeline(prompt).images[0]
image
```

<Tip warning={true}>
تشمل بعض القيود على استخدام LoRAs من Kohya مع 🤗 Diffusers ما يلي:

- قد لا تبدو الصور مثل تلك التي تم إنشاؤها بواسطة واجهات المستخدم - مثل ComfyUI - لأسباب متعددة، والتي تم شرحها [هنا](https://github.com/huggingface/diffusers/pull/4287/#issuecomment-1655110736).
- لا يتم دعم [نقاط تفتيش LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) بالكامل. وتقوم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] بتحميل نقاط تفتيش LyCORIS مع وحدات LoRA وLoCon، ولكن Hada وLoKR غير مدعومتين.
</Tip>

</hfoption>
<hfoption id="TheLastBen">

يعد تحميل نقطة تفتيش من TheLastBen مشابهًا جدًا. على سبيل المثال، لتحميل نقطة تفتيش [TheLastBen/William_Eggleston_Style_SDXL](https://huggingface.co/TheLastBen/William_Eggleston_Style_SDXL):

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("TheLastBen/William_Eggleston_Style_SDXL", weight_name="wegg.safetensors")

# استخدم by william eggleston في المطالبة لتشغيل LoRA
prompt = "a house by william eggleston, sunrays, beautiful, sunlight, sunrays, beautiful"
image = pipeline(prompt=prompt).images[0]
image
```

</hfoption>
</hfoptions>

## محول IP

[محول IP](https://ip-adapter.github.io/) هو محول خفيف الوزن يمكّن المطالبات الصورية لأي نموذج انتشار. ويعمل هذا المحول عن طريق فصل طبقات الاهتمام المتقاطع لميزات الصورة والنص. ويتم تجميد جميع مكونات النموذج الأخرى وتدريب ميزات الصورة المدمجة في UNet فقط. ونتيجة لذلك، عادة ما تكون ملفات محول IP بحجم ~100 ميجابايت فقط.

يمكنك معرفة المزيد حول كيفية استخدام محول IP لمختلف المهام وحالات الاستخدام المحددة في دليل [محول IP](../using-diffusers/ip_adapter).

> [!TIP]
> تدعم Diffusers حاليًا محول IP لبعض خطوط الأنابيب الأكثر شهرة فقط. لا تتردد في فتح طلب ميزة إذا كان لديك حالة استخدام رائعة وتريد دمج محول IP مع خط أنابيب غير مدعوم!
> نقاط تفتيش محول IP الرسمية متوفرة من [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter).

للبدء، قم بتحميل نقطة تفتيش Stable Diffusion.

```py
from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

بعد ذلك، قم بتحميل أوزان محول IP وإضافته إلى خط الأنابيب باستخدام طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`].

```py
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
```

بمجرد التحميل، يمكنك استخدام خط الأنابيب بصورة ومطالبة نصية لتوجيه عملية إنشاء الصورة.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
prompt='best quality, high quality, wearing sunglasses',
ip_adapter_image=image,
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
num_inference_steps=50,
generator=generator,
).images[0]
images
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip-bear.png" />
</div>

### محول IP Plus

يعتمد محول IP على مشفر صورة لتوليد ميزات الصورة. إذا كان مستودع محول IP يحتوي على مجلد `image_encoder` فرعي، يتم تحميل مشفر الصورة تلقائيًا وتسجيله في خط الأنابيب. وإلا، سيتعين عليك تحميل مشفر الصورة بشكل صريح باستخدام نموذج [`~transformers.CLIPVisionModelWithProjection`] وتمريره إلى خط الأنابيب.

وهذا ينطبق على نقاط تفتيش *محول IP Plus* التي تستخدم مشفر الصورة ViT-H.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
"h94/IP-Adapter",
subfolder="models/image_encoder",
torch_dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
image_encoder=image_encoder,
torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
```

### نماذج محول IP Face ID

نماذج محول IP FaceID هي محولات IP تجريبية تستخدم تضمينات الصور التي تم إنشاؤها بواسطة `insightface` بدلاً من تضمينات الصور CLIP. ويستخدم بعض هذه النماذج أيضًا LoRA لتحسين اتساق التعريف.

ويتعين عليك تثبيت `insightface` وجميع متطلباته لاستخدام هذه النماذج.

<Tip warning={true}>
نظرًا لأن النماذج المسبقة التدريب على InsightFace متاحة لأغراض البحث غير التجارية، يتم إصدار نماذج محول IP-FaceID حصريًا لأغراض البحث ولا يُقصد بها الاستخدام التجاري.
</Tip>

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None)
```

إذا كنت تريد استخدام أحد نموذجي محول IP FaceID Plus، فيجب عليك أيضًا تحميل مشفر الصورة CLIP، نظرًا لأن هذه النماذج تستخدم كل من تضمينات الصور `insightface` و CLIP لتحقيق واقعية أفضل.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
"laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5",
image_encoder=image_encoder,
torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plus_sd15.bin")
```