# ملفات النموذج وتخطيطاته

تُحفظ نماذج الانتشار في أنواع ملفات مختلفة وتُنظم بتخطيطات مختلفة. ويخزن برنامج "ديفيوزرز" أوزان النموذج كملفات "سيفتينسورز" في تخطيط "ديفيوزرز-مولتيفولدر"، كما يدعم تحميل الملفات (مثل ملفات "سيفتينسورز" و "كيه بي تي") من تخطيط "سينجل-فايل" الذي يُستخدم بشكل شائع في نظام الانتشار.

يتمتع كل تخطيط بمزايا واستخدامات خاصة به، وسيوضح هذا الدليل كيفية تحميل الملفات والتخطيطات المختلفة، وكيفية تحويلها.

## الملفات

عادةً ما يتم حفظ أوزان نموذج "بايتون تورتش" باستخدام أداة "بيكيل" في Python كملفات "كيه بي تي" أو "بن". ومع ذلك، فإن "بيكيل" غير آمن وقد تحتوي الملفات المؤرشفة على تعليمات برمجية خبيثة يمكن تنفيذها. وهذا الضعف يمثل مصدر قلق خطير بالنظر إلى شعبية مشاركة النماذج. ولمعالجة هذه المشكلة الأمنية، تم تطوير مكتبة "سيفتينسورز" كبديل آمن لـ "بيكيل"، والذي يحفظ النماذج كملفات "سيفتينسورز".

### سيفتينسورز

> [!نصيحة]
> تعرف على المزيد حول قرارات التصميم والسبب في تفضيل ملفات "سيفتينسور" لحفظ وتحميل أوزان النموذج في منشور المدونة "سيفتينسورز المراجعة كآلية آمنة حقًا وتصبح الافتراضية".

"سيفتينسورز" هو تنسيق ملف آمن وسريع لتخزين وتحميل "تينسورز" بشكل آمن. ويقيد "سيفتينسورز" حجم الرأس للحد من أنواع معينة من الهجمات، ويدعم التحميل البطيء (مفيد للإعدادات الموزعة)، ويتميز بسرعة تحميل عامة أسرع.

تأكد من تثبيت مكتبة "سيفتينسورز".

تخزن "سيفتينسورز" الأوزان في ملف "سيفتينسورز". ويحمل "ديفيوزرز" ملفات "سيفتينسورز" بشكل افتراضي إذا كانت متوفرة وكان قد تم تثبيت مكتبة "سيفتينسورز". هناك طريقتان يمكن من خلالهما تنظيم ملفات "سيفتينسورز":

1. تخطيط "ديفيوزرز-مولتيفولدر": قد يكون هناك العديد من ملفات "سيفتينسورز" المنفصلة، واحدة لكل مكون خط أنابيب (مشفّر النص، UNet، VAE)، منظمة في مجلدات فرعية (تفقد مستودع "رانوايإمإل/ستابل-ديفيجن-في1-5" كمثال).
2. تخطيط "سينجل-فايل": قد يتم حفظ جميع أوزان النموذج في ملف واحد (تفقد مستودع "وارريورماما777/أورانجميكس" كمثال).

<hfoptions id="safetensors">
<hfoption id="multifolder">

استخدم طريقة `~DiffusionPipeline.from_pretrained` لتحميل نموذج بملفات "سيفتينسورز" مخزنة في مجلدات متعددة.

```بايثون
من diffusers استيراد DiffusionPipeline

خط الأنابيب = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"،
use_safetensors=True
)
```

</hfoption>
<hfoption id="single file">

استخدم طريقة `~loaders.FromSingleFileMixin.from_single_file` لتحميل نموذج بكل الأوزان المخزنة في ملف "سيفتينسورز" واحد.

```بايثون
من diffusers استيراد StableDiffusionPipeline

خط الأنابيب = StableDiffusionPipeline.from_single_file(
"https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```

</hfoption>
</hfoptions>

#### ملفات لورا

"لورا" هو محول خفيف الوزن سريع وسهل التدريب، مما يجعله شائعًا بشكل خاص لتوليد الصور بطريقة أو نمط معين. وعادة ما يتم تخزين هذه المحولات في ملف "سيفتينسورز"، وهي شائعة على منصات مشاركة النماذج مثل "سيفيتاي".

يتم تحميل "لوراس" في نموذج أساسي باستخدام طريقة `~loaders.LoraLoaderMixin.load_lora_weights`.

```بايثون
من diffusers استيراد StableDiffusionXLPipeline
استيراد الشعلة

# النموذج الأساسي
خط الأنابيب = StableDiffusionXLPipeline.from_pretrained(
"Lykon/dreamshaper-xl-1-0"، torch_dtype=torch.float16، variant="fp16"
).to("cuda")

# تنزيل أوزان لورا
!wget https://civitai.com/api/download/models/168776 -O blueprintify.safetensors

# تحميل أوزان لورا
خط الأنابيب.load_lora_weights("."، weight_name="blueprintify.safetensors")
المطالبة = "bl3uprint، مخطط تفصيلي للغاية لمبنى إمباير ستيت، يوضح كيفية بناء جميع الأجزاء، العديد من النصوص، خلفية شبكة المخطط"
negative_prompt = "lowres، cropped، worst quality، low quality، normal quality، artifacts، signature، watermark، username، blurry، more than one bridge، bad architecture"

الصورة = خط الأنابيب (
prompt=prompt،
negative_prompt=negative_prompt،
generator=torch.manual_seed(0)،
).images[0]
الصورة
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/blueprint-lora.png"/>
</div>

### كيه بي تي

> [!تحذير]
> قد تكون الملفات المؤرشفة غير آمنة لأنها يمكن أن تتعرض للاستغلال لتنفيذ تعليمات برمجية خبيثة. يُنصح باستخدام ملفات "سيفتينسورز" بدلاً من ذلك حيثما أمكن، أو تحويل الأوزان إلى ملفات "سيفتينسورز".

تستخدم وظيفة "تورتش.سيف" في "بايتون تورتش" أداة "بيكيل" في Python لتهيئة النماذج وحفظها. يتم حفظ هذه الملفات كملف "كيه بي تي" وتحتوي على أوزان النموذج بالكامل.

استخدم طريقة `~loaders.FromSingleFileMixin.from_single_file` لتحميل ملف "كيه بي تي" مباشرة.

```بايثون
من diffusers استيراد StableDiffusionPipeline

خط الأنابيب = StableDiffusionPipeline.from_single_file(
"https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
```

## تخطيط التخزين

هناك طريقتان لتنظيم ملفات النموذج، إما في تخطيط "ديفيوزرز-مولتيفولدر" أو في تخطيط "سينجل-فايل". ويكون تخطيط "ديفيوزرز-مولتيفولدر" هو الافتراضي، ويتم تخزين كل ملف مكون (مشفّر النص، UNet، VAE) في مجلد فرعي منفصل. ويدعم "ديفيوزرز" أيضًا تحميل النماذج من تخطيط "سينجل-فايل" حيث يتم تجميع جميع المكونات معًا.
### Diffusers-multifolder

يوفر تخطيط Diffusers-multifolder طريقة تخزين افتراضية لبرامج Diffusers. يتم تخزين أوزان كل مكون (مشفّر النص، UNet، VAE) في مجلد فرعي منفصل. يمكن تخزين الأوزان على شكل ملفات safetensors أو ckpt.

لعرض التحميل من تخطيط Diffusers-multifolder، استخدم طريقة [`~DiffusionPipeline.from_pretrained`] .

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16،
variant="fp16"،
use_safetensors=True,
).to("cuda")
```

تشمل فوائد استخدام تخطيط Diffusers-multifolder ما يلي:

1. أسرع في تحميل كل ملف مكون بشكل فردي أو بالتوازي.
2. تقليل استخدام الذاكرة لأنك تقوم بتحميل المكونات التي تحتاجها فقط. على سبيل المثال، تمتلك النماذج مثل [SDXL Turbo](https://hf.co/stabilityai/sdxl-turbo) و [SDXL Lightning](https://hf.co/ByteDance/SDXL-Lightning) و [Hyper-SD](https://hf.co/ByteDance/Hyper-SD) نفس المكونات باستثناء UNet. يمكنك إعادة استخدام مكوناتها المشتركة مع طريقة [`~DiffusionPipeline.from_pipe`] دون استهلاك أي ذاكرة إضافية (الق نظرة على دليل [إعادة استخدام خط الأنابيب](./loading#reuse-a-pipeline) ) وقم بتحميل UNet فقط. بهذه الطريقة، أنت لست بحاجة إلى تنزيل مكونات مكررة واستخدام المزيد من الذاكرة بشكل غير ضروري.

```py
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler

# قم بتنزيل نموذج واحد
sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"،
torch_dtype=torch.float16،
variant="fp16"،
use_safetensors=True,
).to("cuda")

# قم بالتبديل إلى UNet لنموذج آخر
unet = UNet2DConditionModel.from_pretrained(
"stabilityai/sdxl-turbo"،
subfolder="unet"،
torch_dtype=torch.float16،
variant="fp16"،
use_safetensors=True
)
# إعادة استخدام جميع المكونات نفسها في نموذج جديد باستثناء UNet
turbo_pipeline = StableDiffusionXLPipeline.from_pipe(
sdxl_pipeline, unet=unet,
).to("cuda")
turbo_pipeline.scheduler = EulerDiscreteScheduler.from_config(
turbo_pipeline.scheduler.config,
timestep+spacing="trailing"
)
image = turbo_pipeline(
"رائد فضاء يركب وحيد القرن على المريخ"،
num_inference_steps=1،
guidance_scale=0.0,
).images[0]
image
```

3. تقليل متطلبات التخزين لأنه إذا كان أحد المكونات، مثل VAE SDXL، مشتركًا بين عدة نماذج، فستحتاج فقط إلى تنزيله وتخزين نسخة واحدة منه بدلاً من تنزيله وتخزينه عدة مرات. بالنسبة إلى 10 من نماذج SDXL، يمكن أن يوفر ذلك حوالي 3.5 جيجابايت من مساحة التخزين. وتكون وفورات التخزين أكبر بالنسبة للنماذج الأحدث مثل PixArt Sigma، حيث يبلغ حجم مشفر النص وحده حوالي 19 جيجابايت!

4. المرونة لاستبدال مكون في النموذج بإصدار أحدث أو أفضل.

```py
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix"، torch_dtype=torch.float16، use_safetensors=True)
pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"،
vae=vae,
torch_dtype=torch.float16،
variant="fp16"،
use_safetensors=True,
).to("cuda")
```

5. المزيد من الرؤية والمعلومات حول مكونات النموذج، والتي يتم تخزينها في ملف [config.json](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json) في كل مجلد فرعي للمكونات.

### ملف واحد

يتم في تخطيط الملف الواحد تخزين جميع أوزان النموذج في ملف واحد. يتم الاحتفاظ بأوزان جميع مكونات النموذج (مشفّر النص، UNet، VAE) معًا بدلاً من فصلها في مجلدات فرعية. يمكن أن يكون هذا ملف safetensors أو ckpt.

لعرض التحميل من تخطيط ملف واحد، استخدم طريقة [`~loaders.FromSingleFileMixin.from_single_file`] .

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"،
torch_dtype=torch.float16،
variant="fp16"،
use_safetensors=True,
).to("cuda")
```

تشمل فوائد استخدام تخطيط ملف واحد ما يلي:

1. التوافق السهل مع واجهات الانتشار مثل [ComfyUI](https://github.com/comfyanonymous/ComfyUI) أو [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) والتي تستخدم بشكل شائع تخطيط ملف واحد.

2. من الأسهل إدارة (تنزيل ومشاركة) ملف واحد.

## تحويل التخطيط والملفات

يوفر Diffusers العديد من البرامج النصية والطرق لتحويل تخطيطات التخزين وتنسيقات الملفات لتمكين الدعم الأوسع عبر نظام الانتشار.

الق نظرة على مجموعة [diffusers/scripts](https://github.com/huggingface/diffusers/tree/main/scripts) للعثور على برنامج نصي يناسب احتياجات التحويل الخاصة بك.

> [!TIP]
> تشير البرامج النصية التي تحتوي على "to_diffusers" الملحقة في النهاية إلى أنها تحول نموذجًا إلى تخطيط Diffusers-multifolder. تحتوي كل برنامج نصي على مجموعة محددة من الحجج الخاصة لتكوين التحويل، لذا تأكد من التحقق من الحجج المتاحة!

على سبيل المثال، لتحويل نموذج Stable Diffusion XL المخزن في تخطيط Diffusers-multifolder إلى تخطيط ملف واحد، قم بتشغيل البرنامج النصي [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) . قم بتوفير مسار النموذج الذي تريد تحويله ومسار حفظ النموذج المحول. يمكنك أيضًا تحديد ما إذا كنت تريد حفظ النموذج كملف safetensors وما إذا كنت تريد حفظ النموذج بنصف الدقة.

```bash
python convert_diffusers_to_original_sdxl.py --model_path path/to/model/to/convert --checkpoint_path path/to/save/model/to --use_safetensors
```

يمكنك أيضًا حفظ نموذج إلى تخطيط Diffusers-multifolder باستخدام طريقة [`~DiffusionPipeline.save_pretrained`] . يقوم هذا بإنشاء دليل لك إذا لم يكن موجودًا بالفعل، كما يقوم أيضًا بحفظ الملفات كملف safetensors بشكل افتراضي.

```py
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"،
)
pipeline.save_pretrained()
```

أخيرًا، هناك أيضًا مساحات، مثل [SD To Diffusers](https://hf.co/spaces/diffusers/sd-to-diffusers) و [SD-XL To Diffusers](https://hf.co/spaces/diffusers/sdxl-to-diffusers)، والتي توفر واجهة أكثر ملاءمة للمستخدم لتحويل النماذج إلى تخطيط Diffusers-multifolder. هذا هو أسهل وأكثر الخيارات ملاءمة لتحويل التخطيطات، وسيقوم بفتح PR على مستودع النموذج الخاص بك بالملفات المحولة. ومع ذلك، فإن هذا الخيار ليس موثوقًا به مثل تشغيل برنامج نصي، وقد يفشل المساحة للنماذج الأكثر تعقيدًا.

## استخدام تخطيط ملف واحد

الآن بعد أن تعرفت على الاختلافات بين تخطيط Diffusers-multifolder وتخطيط الملف الواحد، يُظهر لك هذا القسم كيفية تحميل نماذج ومكونات خط الأنابيب، وتخصيص خيارات التكوين للتحميل، وتحميل الملفات المحلية باستخدام طريقة [`~loaders.FromSingleFileMixin.from_single_file`] .

### تحميل خط أنابيب أو نموذج

مرر مسار ملف خط الأنابيب أو النموذج إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`] لتحميله.

<hfoptions id="pipeline-model">
<hfoption id="pipeline">

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path)
```

</hfoption>
<hfoption id="model">

```py
from diffusers import StableCascadeUNet

ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
model = StableCascadeUNet.from_single_file(ckpt_path)
```

</hfoption>
</hfoptions>

قم بتخصيص المكونات في خط الأنابيب عن طريق تمريرها مباشرةً إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`] . على سبيل المثال، يمكنك استخدام جدول زمني مختلف في خط الأنابيب.

```py
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
scheduler = DDIMScheduler()
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler)
```

أو يمكنك استخدام نموذج ControlNet في خط الأنابيب.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipeline = StableDiffusionControlNetPipeline.from_single_file(ckpt_path, controlnet=controlnet)
```
### تخصيص خيارات التهيئة

تمتلك النماذج ملف تهيئة يحدد سماتها مثل عدد المدخلات في شبكة UNet. خيارات تهيئة الأنابيب متاحة في فئة الأنبوب. على سبيل المثال، إذا نظرت إلى فئة [`StableDiffusionXLInstructPix2PixPipeline`]()، فهناك خيار لقياس الصورة الكامنة مع معامل `is_cosxl_edit`.

يمكن العثور على ملفات التهيئة هذه في مستودع Hub النموذجي أو في موقع آخر الذي نشأ منه ملف التهيئة (على سبيل المثال، مستودع GitHub أو محليًا على جهازك).

<hfoptions id="config-file">

<hfoption id="Hub configuration file">

> [!TIP]
> طريقة [`~loaders.FromSingleFileMixin.from_single_file`]() تقوم تلقائيًا بتعيين نقطة التفتيش إلى مستودع النموذج المناسب، ولكن هناك حالات يكون من المفيد فيها استخدام معامل "التهيئة". على سبيل المثال، إذا كانت مكونات النموذج في نقطة التفتيش مختلفة عن نقطة التفتيش الأصلية أو إذا لم يكن لنقطة التفتيش البيانات الوصفية اللازمة لتحديد تهيئة الأنبوب بشكل صحيح.

طريقة [`~loaders.FromSingleFileMixin.from_single_file`]() تحدد تلقائيًا التهيئة التي سيتم استخدامها من ملف التهيئة في مستودع النماذج. يمكنك أيضًا تحديد التهيئة التي سيتم استخدامها صراحةً من خلال توفير معرف المستودع لمعامل "التهيئة".

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/segmind/SSD-1B/blob/main/SSD-1B.safetensors"
repo_id = "segmind/SSD-1B"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, config=repo_id)
```

يتم تحميل النموذج لملف تهيئة [UNet]() و [VAE]() و [encoder النصي]() من مجلداتهم الفرعية الخاصة في المستودع.

</hfoption>

<hfoption id="original configuration file">

يمكن أيضًا لطريقة [`~loaders.FromSingleFileMixin.from_single_file`]() تحميل ملف تهيئة الأصلي للأنبوب المخزن في مكان آخر. قم بتمرير مسار محلي أو عنوان URL لملف تهيئة الأصلي إلى معامل `original_config`.

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
original_config = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, original_config=original_config)
```

> [!TIP]
> تحاول تطبيقات Diffusers استنتاج مكونات الأنبوب بناءً على توقيعات النوع لفئة الأنبوب عند استخدام `original_config` مع `local_files_only=True`، بدلاً من استرداد ملفات التهيئة من مستودع النموذج على Hub. يمنع هذا التغييرات التراجعية في التعليمات البرمجية التي لا يمكنها الاتصال بالإنترنت لاسترداد ملفات التهيئة اللازمة.
>
> هذا ليس موثوقًا مثل توفير مسار إلى مستودع نموذج محلي مع معامل "التهيئة"، وقد يؤدي إلى أخطاء أثناء تهيئة الأنبوب. لتجنب الأخطاء، قم بتشغيل الأنبوب مع `local_files_only=False` مرة واحدة لتحميل ملفات تهيئة الأنبوب المناسبة إلى ذاكرة التخزين المؤقت المحلية.

</hfoption>

</hfoptions>

في حين أن ملفات التهيئة تحدد الافتراضيات الافتراضية للأنبوب أو النماذج، يمكنك تجاوزها من خلال توفير المعلمات مباشرةً إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`]()، يمكن تهيئة أي معلمة مدعومة بواسطة فئة النموذج أو الأنبوب بهذه الطريقة.

<hfoptions id="override">

<hfoption id="pipeline">

على سبيل المثال، لقياس الصورة الكامنة في [`StableDiffusionXLInstructPix2PixPipeline`]()، قم بتمرير معامل `is_cosxl_edit`.

```python
from diffusers import StableDiffusionXLInstructPix2PixPipeline

ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(ckpt_path, config="diffusers/sdxl-instructpix2pix-768", is_cosxl_edit=True)
```

</hfoption>

<hfoption id="model">

على سبيل المثال، لزيادة أبعاد الاهتمام في [`UNet2DConditionModel`]()، قم بتمرير معامل `upcast_attention`.

```python
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, upcast_attention=True)
```

</hfoption>

</hfoptions>

### الملفات المحلية

في Diffusers>=v0.28.0، تحاول طريقة [`~loaders.FromSingleFileMixin.from_single_file`]() تهيئة أنبوب أو نموذج من خلال استنتاج نوع النموذج من المفاتيح في ملف نقطة التفتيش. يتم استخدام نوع النموذج المستنتج لتحديد مستودع النموذج المناسب على Hugging Face Hub لتهيئة النموذج أو الأنبوب.

على سبيل المثال، ستستخدم أي نقطة تفتيش ملف واحد تستند إلى نموذج Stable Diffusion XL الأساسي مستودع النموذج [stabilityai/stable-diffusion-xl-base-1.0]() لتهيئة الأنبوب.

ولكن إذا كنت تعمل في بيئة ذات إمكانية وصول مقيدة إلى الإنترنت، فيجب عليك تنزيل ملفات التهيئة باستخدام وظيفة [`~huggingface_hub.snapshot_download`]()، ونقطة تفتيش النموذج باستخدام وظيفة [`~huggingface_hub.hf_hub_download`]()، يتم تنزيل هذه الملفات افتراضيًا إلى دليل ذاكرة التخزين المؤقت لـ Hugging Face Hub، ولكن يمكنك تحديد دليل مفضل لتنزيل الملفات إليه باستخدام معامل `local_dir`.

قم بتمرير مسارات التهيئة ونقطة التفتيش إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`]() لتحميلها محليًا.

<hfoptions id="local">

<hfoption id="Hub cache directory">

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
repo_id="segmind/SSD-1B",
filename="SSD-1B.safetensors"
)

my_local_config_path = snapshot_download(
repo_id="segmind/SSD-1B",
allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)

pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```

</hfoption>

<hfoption id="specific local directory">

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
repo_id="segmind/SSD-1B",
filename="SSD-1B.safetensors"
local_dir="my_local_checkpoints"
)

my_local_config_path = snapshot_download(
repo_id="segmind/SSD-1B",
allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
local_dir="my_local_config"
)

pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```

</hfoption>

</hfoptions>

#### الملفات المحلية بدون symlink

> [!TIP]
> في huggingface_hub>=v0.23.0، حجة `local_dir_use_symlinks` ليست ضرورية لوظائف [`~huggingface_hub.hf_hub_download`]() و [`~huggingface_hub.snapshot_download`]()، تعتمد طريقة [`~loaders.FromSingleFileMixin.from_single_file`]() على آلية التخزين المؤقت لـ [huggingface_hub]() لاسترداد وتخزين نقاط التفتيش وملفات التهيئة للنماذج والأنابيب، إذا كنت تعمل مع نظام ملفات لا يدعم إنشاء الارتباطات الرمزية، فيجب عليك أولاً تنزيل ملف نقطة التفتيش إلى دليل محلي وتعطيل الارتباطات الرمزية باستخدام معامل `local_dir_use_symlink=False` في وظائف [`~huggingface_hub.hf_hub_download`]() و [`~huggingface_hub.snapshot_download`]()،

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
repo_id="segmind/SSD-1B",
filename="SSD-1B.safetensors"
local_dir="my_local_checkpoints",
local_dir_use_symlinks=False
)
print("My local checkpoint: ", my_local_checkpoint_path)

my_local_config_path = snapshot_download(
repo_id="segmind/SSD-1B",
allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
local_dir_use_symlinks=False,
)
print("My local config: ", my_local_config_path)

```

بعد ذلك، يمكنك تمرير المسارات المحلية إلى معلمات `pretrained_model_link_or_path` و`config`.

```python
pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```