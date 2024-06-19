# النص إلى الصورة

عند التفكير في نماذج الانتشار، فإن النص إلى الصورة هو عادة أحد الأشياء الأولى التي تتبادر إلى الذهن. ينشئ النص إلى الصورة صورة من وصف نصي (على سبيل المثال، "رائد فضاء في غابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8K") والتي يُشار إليها أيضًا باسم *سؤال*.

من مستوى عالٍ جدًا، يأخذ نموذج الانتشار سؤالاً وبعض الضوضاء الأولية العشوائية، ويزيل الضوضاء بشكل تكراري لبناء صورة. وتوجه عملية "إزالة التشويش" بواسطة السؤال، وبمجرد انتهاء عملية إزالة التشويش بعد عدد محدد مسبقًا من الخطوات الزمنية، يتم فك ترميز تمثيل الصورة إلى صورة.

يمكنك إنشاء الصور من سؤال في 🤗 Diffusers في خطوتين:

1. قم بتحميل نقطة تفتيش في فئة [`AutoPipelineForText2Image`]، والتي تقوم تلقائيًا بالكشف عن فئة الأنابيب المناسبة لاستخدامها بناءً على نقطة التفتيش:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
```

2. قم بتمرير سؤال إلى الأنابيب لإنشاء صورة:

```py
image = pipeline(
"زجاج ملون لدارث فيدر، إضاءة خلفية، تكوين مركزي، تحفة فنية، صور واقعية، 8K"
).images[0]
image
```

## النماذج الشائعة

أكثر نماذج النص إلى الصورة شيوعًا هي [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)، و[Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)، و[Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder). هناك أيضًا نماذج ControlNet أو المحولات التي يمكن استخدامها مع نماذج النص إلى الصورة للتحكم بشكل أكثر مباشرة في إنشاء الصور. تختلف النتائج من كل نموذج اختلافًا طفيفًا بسبب هندستها وعملية التدريب الخاصة بها، ولكن بغض النظر عن النموذج الذي تختاره، فإن استخدامها متماثل إلى حد ما. دعونا نستخدم نفس السؤال لكل نموذج ونقارن نتائجها.

### Stable Diffusion v1.5

[Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) هو نموذج انتشار الكامن المبدئي من [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)، وتم ضبط دقته لـ 595K خطوة على صور 512x512 من مجموعة بيانات LAION-Aesthetics V2. يمكنك استخدام هذا النموذج مثل:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8K"، generator=generator).images[0]
image
```

### Stable Diffusion XL

SDXL هو إصدار أكبر بكثير من نماذج Stable Diffusion السابقة، ويتضمن عملية نموذج من مرحلتين تضيف المزيد من التفاصيل إلى الصورة. كما يتضمن بعض "التكييفات الدقيقة" الإضافية لتوليد صور عالية الجودة تتمحور حول الموضوعات. الق نظرة على دليل [SDXL](sdxl) الأكثر شمولاً لمعرفة المزيد حول كيفية استخدامه. بشكل عام، يمكنك استخدام SDXL مثل:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8K"، generator=generator).images[0]
image
```

### Kandinsky 2.2

يختلف نموذج Kandinsky قليلاً عن نماذج Stable Diffusion لأنه يستخدم أيضًا نموذج أولوية الصورة لإنشاء تضمينات يتم استخدامها لتحسين محاذاة النص والصور في نموذج الانتشار.

أسهل طريقة لاستخدام Kandinsky 2.2 هي:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8K"، generator=generator).images[0]
image
```

### ControlNet

ControlNet هي نماذج أو محولات مساعدة يتم ضبط دقتها على قمة نماذج النص إلى الصورة، مثل [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). يوفر استخدام نماذج ControlNet بالاقتران مع نماذج النص إلى الصورة خيارات متنوعة للتحكم بشكل أكثر صراحة في كيفية إنشاء صورة. باستخدام ControlNet، يمكنك إضافة إدخال شرطي صورة إضافية إلى النموذج. على سبيل المثال، إذا قدمت صورة لوضع إنسان (عادة ما يتم تمثيلها على شكل نقاط رئيسية متعددة يتم توصيلها إلى هيكل عظمي) كإدخال شرطي، فإن النموذج ينشئ صورة تتبع وضع الصورة. تحقق من دليل [ControlNet](controlnet) الأكثر عمقًا لمعرفة المزيد حول إدخالات الشرطية الأخرى وكيفية استخدامها.

في هذا المثال، دعنا نشترط على ControlNet بصورة تقدير وضع الإنسان. قم بتحميل نموذج ControlNet الذي تم ضبط دقته مسبقًا على تقديرات الوضع البشري:

```py
from diffusers import ControlNetModel, AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained(
"lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pose_image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/control.png")
```

مرر `controlnet` إلى [`AutoPipelineForText2Image`]، وقدم السؤال وصورة تقدير الوضع:

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8K"، image=pose_image، generator=generator).images[0]
image
```

## تكوين معلمات الأنابيب

هناك عدد من المعلمات التي يمكن تكوينها في الأنابيب والتي تؤثر على كيفية إنشاء صورة. يمكنك تغيير حجم الصورة الناتجة، وتحديد سؤال سلبي لتحسين جودة الصورة، والمزيد. يتعمق هذا القسم في كيفية استخدام هذه المعلمات.

### الارتفاع والعرض

تحكم معلمات "الارتفاع" و"العرض" في ارتفاع وعرض الصورة المولدة (بالبكسل). بشكل افتراضي، ينتج نموذج Stable Diffusion v1.5 صورًا بحجم 512x512، ولكن يمكنك تغيير ذلك إلى أي حجم يكون مضاعفًا لـ 8. على سبيل المثال، لإنشاء صورة مستطيلة:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
image = pipeline(
"رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8K"، height=768، width=512
).images[0]
image
```

<Tip warning={true}>
قد يكون للنماذج الأخرى أحجام صور افتراضية مختلفة اعتمادًا على أحجام الصور في مجموعة بيانات التدريب. على سبيل المثال، الحجم الافتراضي لصورة SDXL هو 1024x1024 وقد يؤدي استخدام قيم "ارتفاع" و"عرض" أقل إلى انخفاض جودة الصورة. تأكد من التحقق من مرجع API للنموذج أولاً!
</Tip>
### مقياس التوجيه

تؤثر قيمة معلمة `guidance_scale` على مدى تأثير التوجيه على توليد الصورة. تعطي القيمة المنخفضة النموذج "الإبداع" لتوليد صور لها علاقة أكثر مرونة بالتوجيه. تدفع قيم `guidance_scale` الأعلى النموذج إلى اتباع التوجيه عن كثب، وإذا كانت هذه القيمة مرتفعة للغاية، فقد تلاحظ بعض العيوب في الصورة المولدة.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
image = pipeline(
"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", guidance_scale=3.5
).images[0]
image
```

### توجيه سلبي

تماماً كما يقوم التوجيه بتوجيه التوليد، فإن التوجيه *السلبي* يوجه النموذج بعيدًا عن الأشياء التي لا تريدها أن يقوم النموذج بتوليدها. ويُستخدم هذا عادة لتحسين جودة الصورة العامة عن طريق إزالة ميزات الصورة السيئة أو الرديئة مثل "دقة منخفضة" أو "تفاصيل سيئة". يمكنك أيضًا استخدام توجيه سلبي لإزالة محتوى الصورة أو تعديله أو تغيير أسلوبها.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
image = pipeline(
prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
).images[0]
image
```

### المولد

تمكّن [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#generator) قابلية إعادة الإنتاج في خط الأنابيب عن طريق تعيين بذرة يدوية. يمكنك استخدام `Generator` لإنشاء دفعات من الصور وتحسين صورة تم إنشاؤها من بذرة بشكل تكراري كما هو موضح بالتفصيل في دليل [تحسين جودة الصورة باستخدام التوليد المحدد](reusing_seeds).

يمكنك تعيين بذرة و`Generator` كما هو موضح أدناه. يجب أن يؤدي إنشاء صورة باستخدام `Generator` إلى إرجاع نفس النتيجة في كل مرة بدلاً من إنشاء صورة جديدة بشكل عشوائي.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(30)
image = pipeline(
"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
generator=generator,
).images[0]
image
```

## التحكم في توليد الصور

هناك طرق عديدة لممارسة المزيد من التحكم في كيفية توليد صورة خارج نطاق تكوين معلمات خط الأنابيب، مثل وزن التوجيه ونماذج ControlNet.

### وزن التوجيه

وزن التوجيه هو تقنية لزيادة أو تقليل أهمية المفاهيم في توجيه ما لتأكيد أو تقليل ميزات معينة في صورة. نوصي باستخدام مكتبة [Compel](https://github.com/damian0815/compel) لمساعدتك في إنشاء تضمين التوجيه المرجح.

<Tip>
تعرف على كيفية إنشاء تضمينات التوجيه في دليل [وزن التوجيه](weighted_prompts). يركز هذا المثال على كيفية استخدام تضمينات التوجيه في خط الأنابيب.
</Tip>

بمجرد إنشاء التضمينات، يمكنك تمريرها إلى معلمة `prompt_embeds` (و`negative_prompt_embeds` إذا كنت تستخدم توجيهًا سلبيًا) في خط الأنابيب.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
image = pipeline(
prompt_embeds=prompt_embeds, # generated from Compel
negative_prompt_embeds=negative_prompt_embeds, # generated from Compel
).images[0]
```

### ControlNet

كما رأيت في قسم [ControlNet](#controlnet)، توفر هذه النماذج طريقة أكثر مرونة ودقة لتوليد الصور من خلال دمج إدخال صورة شرطية إضافية. يتم تدريب كل نموذج ControlNet مسبقًا على نوع معين من صور الشرطية لتوليد صور جديدة تشبهها. على سبيل المثال، إذا كان لديك نموذج ControlNet تم تدريبه مسبقًا على خرائط العمق، فيمكنك إعطاء النموذج خريطة عمق كإدخال شرطي وسينشئ صورة تحافظ على المعلومات المكانية فيها. هذا أسرع وأسهل من تحديد معلومات العمق في توجيه. يمكنك حتى دمج إدخالات شرطية متعددة مع [MultiControlNet](controlnet#multicontrolnet)!

هناك العديد من أنواع الإدخالات الشرطية التي يمكنك استخدامها، ويدعم 🤗 Diffusers ControlNet لنماذج Stable Diffusion وSDXL. اطلع على دليل [ControlNet](controlnet) الأكثر شمولاً لمعرفة كيفية استخدام هذه النماذج.

## التحسين

تعد نماذج الانتشار كبيرة، وطبيعة إزالة الضوضاء التكرارية للصورة مكلفة وكثيفة الحساب. ولكن هذا لا يعني أنك بحاجة إلى الوصول إلى وحدات معالجة الرسومات (GPU) القوية - أو حتى العديد منها - لاستخدامها. هناك العديد من تقنيات التحسين لتشغيل نماذج الانتشار على موارد المستهلك والطبقة المجانية. على سبيل المثال، يمكنك تحميل أوزان النموذج بدقة نصفية لتوفير ذاكرة GPU وزيادة السرعة أو نقل النموذج بالكامل إلى وحدة معالجة الرسومات لتوفير المزيد من الذاكرة.

يدعم PyTorch 2.0 أيضًا آلية اهتمام أكثر كفاءة في الذاكرة تسمى [*scaled dot product attention*](../optimization/torch2.0#scaled-dot-product-attention) يتم تمكينها تلقائيًا إذا كنت تستخدم PyTorch 2.0. يمكنك دمج هذا مع [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) لتسريع الكود الخاص بك أكثر:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```

للحصول على مزيد من النصائح حول كيفية تحسين الكود الخاص بك لتوفير الذاكرة وتسريع الاستدلال، اقرأ دليلي [Memory and speed](../optimization/fp16) و [Torch 2.0](../optimization/torch2.0).