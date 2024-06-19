# DreamBooth  

تقنية تدريب تقوم بتحديث نموذج الانتشار بالكامل من خلال التدريب على عدد قليل فقط من الصور للموضوع أو الأسلوب. تعمل من خلال ربط كلمة خاصة في المطالبة بالصور المثالية.

إذا كنت تتدرب على وحدة معالجة رسومات (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين المعلمات "gradient_checkpointing" و"mixed_precision" في أمر التدريب. يمكنك أيضًا تقليل البصمة الخاصة بك باستخدام الاهتمام بكفاءة الذاكرة مع xFormers. يتم أيضًا دعم التدريب JAX/Flax للتدريب الفعال على وحدات معالجة الرسوميات (TPUs) ووحدات معالجة الرسوميات (GPUs)، ولكنه لا يدعم نقاط التفتيش التدرجية أو xFormers. يجب أن يكون لديك وحدة معالجة رسومات (GPU) بها أكثر من 30 جيجابايت من الذاكرة إذا كنت تريد التدريب بشكل أسرع باستخدام Flax.

سيتعمق هذا الدليل في برنامج النص البرمجي "train_dreambooth.py" لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

انتقل إلى مجلد المثال باستخدام برنامج النص البرمجي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

🤗 Accelerate عبارة عن مكتبة للمساعدة في التدريب على وحدات معالجة الرسوميات (GPU) / وحدات معالجة الرسوميات (TPU) متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين برنامج الإعداد التدريبي الخاص بك بناءً على الأجهزة وبيئة العمل لديك. الق نظرة على جولة سريعة في 🤗 Accelerate لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل إنشاء مجموعة بيانات للتدريب لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع برنامج النص البرمجي للتدريب.

تسلط الأقسام التالية الضوء على أجزاء من برنامج النص البرمجي للتدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي ودعنا نعرف إذا كان لديك أي أسئلة أو مخاوف.

## معلمات البرنامج النصي

DreamBooth حساس جدًا لمعلمات التدريب، ومن السهل أن يحدث بها إفراط في التكييف. اقرأ منشور المدونة "تدريب Stable Diffusion باستخدام Dreambooth باستخدام Diffusers" للإعدادات الموصى بها لموضوعات مختلفة لمساعدتك في اختيار المعلمات المناسبة.

يقدم برنامج النص البرمجي للتدريب العديد من المعلمات لتخصيص تشغيل التدريب الخاص بك. يمكن العثور على جميع المعلمات ووصفاتها في دالة "parse_args()". يتم تعيين المعلمات بقيم افتراضية يجب أن تعمل بشكل جيد خارج الصندوق، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت ترغب في ذلك.

على سبيل المثال، للتدريب بتنسيق bf16:

بعض المعلمات الأساسية والمهمة التي يجب معرفتها وتحديدها هي:

- pretrained_model_name_or_path: اسم النموذج على Hub أو مسار محلي للنموذج الذي تم تدريبه مسبقًا
- instance_data_dir: المسار إلى المجلد الذي يحتوي على مجموعة بيانات التدريب (صور المثال)
- instance_prompt: مطالبة النص التي تحتوي على الكلمة الخاصة لصور المثال
- train_text_encoder: ما إذا كان سيتم أيضًا تدريب مشفر النص
- output_dir: المكان الذي سيتم فيه حفظ النموذج الذي تم تدريبه
- push_to_hub: ما إذا كان سيتم دفع النموذج الذي تم تدريبه إلى Hub
- checkpointing_steps: تكرار حفظ نقطة تفتيش أثناء تدريب النموذج؛ هذا مفيد إذا تم مقاطعة التدريب لسبب ما، فيمكنك الاستمرار في التدريب من تلك النقطة عن طريق إضافة --resume_from_checkpoint إلى أمر التدريب الخاص بك

### وزن الحد الأدنى من نسبة الإشارة إلى الضوضاء

يمكن أن تساعد استراتيجية وزن الحد الأدنى من نسبة الإشارة إلى الضوضاء (Min-SNR) في التدريب من خلال إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم برنامج النص البرمجي للتدريب التنبؤ بـ "epsilon" (الضوضاء) أو "v_prediction"، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. استراتيجية الترجيح هذه مدعومة فقط بواسطة PyTorch وغير متوفرة في برنامج النص البرمجي للتدريب Flax.

أضف المعلمة --snr_gamma وقم بتعيينها على القيمة الموصى بها 5.0:

### خسارة الحفاظ على الأولوية

خسارة الحفاظ على الأولوية هي طريقة تستخدم عينات مولدة من النموذج نفسه لمساعدته على تعلم كيفية إنشاء صور أكثر تنوعًا. لأن صور العينة المولدة هذه تنتمي إلى نفس الفئة التي قدمتها، فإنها تساعد النموذج على الاحتفاظ بما تعلمه عن الفئة وكيف يمكنه استخدام ما يعرفه بالفعل عن الفئة لإجراء تكوينات جديدة.

- with_prior_preservation: ما إذا كان سيتم استخدام خسارة الحفاظ على الأولوية
- prior_loss_weight: يتحكم في تأثير خسارة الحفاظ على الأولوية على النموذج
- class_data_dir: المسار إلى المجلد الذي يحتوي على صور العينة المولدة من الفئة
- class_prompt: مطالبة النص التي تصف فئة صور العينة المولدة

### تدريب مشفر النص

لتحسين جودة المخرجات المولدة، يمكنك أيضًا تدريب مشفر النص بالإضافة إلى UNet. يتطلب ذلك ذاكرة إضافية وتحتاج إلى وحدة معالجة رسومات (GPU) بها 24 جيجابايت على الأقل من ذاكرة الوصول العشوائي (VRAM). إذا كان لديك الأجهزة اللازمة، فإن تدريب مشفر النص ينتج نتائج أفضل، خاصة عند إنشاء صور الوجوه. قم بتمكين هذا الخيار عن طريق:
يأتي DreamBooth مع فئات مجموعات البيانات الخاصة به:

- `DreamBoothDataset`: يقوم بمعالجة الصور وصور الفئات، ويقوم برمجة المطالبات للتدريب.
- `PromptDataset`: يقوم بتوليد تضمين المطالبة لتوليد صور الفئة.

إذا قمت بتمكين "خسارة الحفاظ على الأولوية"، يتم إنشاء صور الفئة هنا:

```py
sample_dataset = PromptDataset(args.class_prompt, num_new_images)
sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

sample_dataloader = accelerator.prepare(sample_dataloader)
pipeline.to(accelerator.device)

for example in tqdm(
sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
):
images = pipeline(example["prompt"]).images
```

بعد ذلك، تأتي دالة `main()` التي تتولى إعداد مجموعة البيانات للتدريب وحلقة التدريب نفسها. يقوم النص البرمجي بتحميل "مُرمِّز الرموز" (tokenizer) والمجدول والنماذج:

```py
# تحميل مُرمِّز الرموز
if args.tokenizer_name:
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
tokenizer = AutoTokenizer.from_pretrained(
args.pretrained_model_name_or_path,
subfolder="tokenizer",
revision=args.revision,
use_fast=False,
)

# تحميل المُجدول والنماذج
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)

if model_has_vae(args):
vae = AutoencoderKL.from_pretrained(
args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
)
else:
vae = None

unet = UNet2DConditionModel.from_pretrained(
args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

بعد ذلك، حان الوقت لإنشاء مجموعة بيانات التدريب وDataLoader من `DreamBoothDataset`:

```py
train_dataset = DreamBoothDataset(
instance_data_root=args.instance_data_dir,
instance_prompt=args.instance_prompt,
class_data_root=args.class_data_dir if args.with_prior_preservation else None,
class_prompt=args.class_prompt,
class_num=args.num_class_images,
tokenizer=tokenizer,
size=args.resolution,
center_crop=args.center_crop,
encoder_hidden_states=pre_computed_encoder_hidden_states,
class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
tokenizer_max_length=args.tokenizer_max_length,
)

train_dataloader = torch.utils.data.DataLoader(
train_dataset,
batch_size=args.train_batch_size,
shuffle=True,
collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
num_workers=args.dataloader_num_workers,
)
```

أخيرًا، تتولى حلقة التدريب الخطوات المتبقية مثل تحويل الصور إلى مساحة خفية، وإضافة الضوضاء إلى الإدخال، والتنبؤ ببقايا الضوضاء، وحساب الخسارة.

إذا كنت تريد معرفة المزيد عن كيفية عمل حلقة التدريب، تحقق من البرنامج التعليمي "فهم خطوط الأنابيب والنماذج والمُجدولات" الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.
## تشغيل البرنامج النصي

الآن أنت مستعد لتشغيل برنامج التدريب النصي! 🚀

لأغراض هذا الدليل، ستقوم بتنزيل بعض الصور لـ [كلب](https://huggingface.co/datasets/diffusers/dog-example) وحفظها في دليل. ولكن تذكر، يمكنك إنشاء واستخدام مجموعة البيانات الخاصة بك إذا أردت (راجع الدليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

قم بتعيين متغير البيئة `MODEL_NAME` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي، و`INSTANCE_DIR` إلى المسار الذي قمت بتنزيل صور الكلب إليه للتو، و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه. ستستخدم "sks" ككلمة خاصة لربط التدريب بها.

إذا كنت مهتمًا بمتابعة عملية التدريب، فيمكنك حفظ الصور المولدة بشكل دوري أثناء تقدم التدريب. أضف المعلمات التالية إلى أمر التدريب:

```bash
--validation_prompt="a photo of a sks dog"
--num_validation_images=4
--validation_steps=100
```

قبل تشغيل البرنامج النصي! اعتمادًا على وحدة معالجة الرسومات (GPU) التي لديك، قد تحتاج إلى تمكين بعض التحسينات لتدريب DreamBooth.

<hfoptions id="gpu-select">

<hfoption id="16GB">

على GPU بسعة 16 جيجابايت، يمكنك استخدام محسن bitsandbytes 8-bit ومحطات التدرج لتدريب نموذج DreamBooth. قم بتثبيت bitsandbytes:

```py
pip install bitsandbytes
```

بعد ذلك، أضف المعلمة التالية إلى أمر التدريب:

```bash
accelerate launch train_dreambooth.py \
--gradient_checkpointing \
--use_8bit_adam \
```

</hfoption>

<hfoption id="12GB">

على GPU بسعة 12 جيجابايت، ستحتاج إلى محسن bitsandbytes 8-bit، ومحطات التدرج، وxFormers، وتعيين التدرجات إلى `None` بدلاً من الصفر لتقليل استخدام الذاكرة.

```bash
accelerate launch train_dreambooth.py \
--use_8bit_adam \
--gradient_checkpointing \
--enable_xformers_memory_efficient_attention \
--set_grads_to_none \
```

</hfoption>

<hfoption id="8GB">

على GPU بسعة 8 جيجابايت، ستحتاج إلى [DeepSpeed](https://www.deepspeed.ai/) لنقل بعض المنسوجات من ذاكرة الوصول العشوائي للرسومات (VRAM) إلى وحدة المعالجة المركزية (CPU) أو NVME للسماح بالتدريب باستخدام ذاكرة GPU أقل.

قم بتشغيل الأمر التالي لتكوين بيئة 🤗 Accelerate:

```bash
accelerate config
```

خلال التكوين، تأكد من أنك تريد استخدام DeepSpeed. الآن يجب أن يكون من الممكن التدريب على أقل من 8 جيجابايت من ذاكرة الوصول العشوائي باستخدام DeepSpeed المرحلة 2 والدقة المختلطة fp16 ونقل معلمات النموذج وحالة المحسن إلى وحدة المعالجة المركزية. العيب هو أن هذا يتطلب المزيد من ذاكرة الوصول العشوائي للنظام (~25 جيجابايت). راجع وثائق DeepSpeed للحصول على خيارات تكوين إضافية.

يجب عليك أيضًا تغيير محسن Adam الافتراضي إلى الإصدار الأمثل لـ DeepSpeed من Adam [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu) للحصول على تسريع كبير. يتطلب تمكين `DeepSpeedCPUAdam` أن تكون نسخة مجموعة أدوات CUDA في نظامك هي نفسها المثبتة مع PyTorch.

لا يبدو أن محسنات 8 بت bitsandbytes متوافقة مع DeepSpeed في الوقت الحالي.

هذا كل شيء! لا تحتاج إلى إضافة أي معلمات إضافية إلى أمر التدريب الخاص بك.

</hfoption>

</hfoptions>

<hfoptions id="training-inference">

<hfoption id="PyTorch">

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a photo of sks dog" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_
</hfoption>

<hfoption id="Flax">

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a photo of sks dog" \
--resolution=512 \
--train_batch_size=1 \
--learning_rate=5e-6 \
--max_train_steps=400 \
--push_to_hub
```

</hfoption>

</hfoptions>

بمجرد اكتمال التدريب، يمكنك استخدام نموذجك المدرب حديثًا للاستنتاج!

<Tip>

هل لا تستطيع الانتظار لتجربة نموذجك للاستنتاج قبل اكتمال التدريب؟ 🤭 تأكد من تثبيت أحدث إصدار من 🤗 Accelerate.

```py
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("path/to/model/checkpoint-100/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("path/to/model/checkpoint-100/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، unet=unet، text_encoder=text_encoder، dtype=torch.float16
).to("cuda")

الصورة = pipeline ("A photo of sks dog in a bucket"، num_inference_steps=50، guidance_scale=7.5).images [0]
image.save ("dog-bucket.png")
```

</Tip>

<hfoptions id="training-inference">

<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model"، torch_dtype=torch.float16، use_safetensors=True).to("cuda")
image = pipeline ("A photo of sks dog in a bucket"، num_inference_steps=50، guidance_scale=7.5).images [0]
image.save ("dog-bucket.png")
```

</hfoption>

<hfoption id="Flax">

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("path-to-your-trained-model"، dtype=jax.numpy.bfloat16)

prompt = "A photo of sks dog in a bucket"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed، jax.device_count())
prompt_ids = shard(prompt_ids)

الصور = pipeline (prompt_ids، params، prng_seed، num_inference_steps، jit=True).images
الصور = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples،) + images.shape [-3:])))
image.save ("dog-bucket.png")
```

</hfoption>

</hfoptions>

## لورا

لورا هي تقنية تدريب لخفض عدد المعلمات القابلة للتدريب بشكل كبير. ونتيجة لذلك، يكون التدريب أسرع ويكون من الأسهل تخزين الأوزان الناتجة لأنها أصغر بكثير (~ 100 ميجابايت). استخدم البرنامج النصي [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) لتدريب لورا.

يتم مناقشة برنامج النصي للتدريب لورا بالتفصيل في دليل [تدريب لورا](لورا).

## انتشار مستقر XL

Stable Diffusion XL (SDXL) هو نموذج نصي إلى صورة قوي يولد صور عالية الدقة، ويضيف مشفر نص ثانٍ إلى بنائه المعماري. استخدم البرنامج النصي [train_dreambooth_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py) لتدريب نموذج SDXL باستخدام لورا.

يتم مناقشة برنامج النصي لتدريب SDXL بالتفصيل في دليل [تدريب SDXL](sdxl).
## DeepFloyd IF

نموذج DeepFloyd IF هو نموذج تسريب بكسل متتالي بثلاث مراحل. تقوم المرحلة الأولى بتوليد صورة أساسية، وتقوم المرحلتان الثانية والثالثة تدريجياً بزيادة دقة الصورة الأساسية إلى صورة عالية الدقة بدقة 1024x1024. استخدم النصوص [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) أو [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) لتدريب نموذج DeepFloyd IF باستخدام LoRA أو النموذج الكامل.

يستخدم DeepFloyd IF التباين المتوقع، ولكن نصوص تدريب Diffusers تستخدم الخطأ المتوقع، لذلك يتم تبديل النماذج المدربة من DeepFloyd IF إلى جدول تباين ثابت. ستعمل نصوص التدريب على تحديث تكوين جدول مواعيد النموذج المدرب بالكامل نيابة عنك. ومع ذلك، عند تحميل أوزان LoRA المحفوظة، يجب أيضًا تحديث تكوين جدول مواعيد الأنبوب.

المرحلة 2 من النموذج تتطلب صور تحقق إضافية لزيادة الدقة. يمكنك تنزيل واستخدام نسخة مصغرة من صور التدريب لهذا الغرض.

تقدم عينات التعليمات البرمجية أدناه نظرة عامة موجزة حول كيفية تدريب نموذج DeepFloyd IF باستخدام مزيج من DreamBooth وLoRA. فيما يلي بعض المعلمات المهمة التي يجب ملاحظتها:

* `--resolution=64`، مطلوب دقة أصغر بكثير لأن DeepFloyd IF هو نموذج تسريب بكسل، وللعمل على البكسلات غير المضغوطة، يجب أن تكون صور الإدخال أصغر
* `--pre_compute_text_embeddings`، احسب تضمين النص مسبقًا لتوفير الذاكرة لأن [`~transformers.T5Model`] يمكن أن يستهلك الكثير من الذاكرة
* `--tokenizer_max_length=77`، يمكنك استخدام طول نص افتراضي أطول مع T5 كمشفر نص، ولكن إجراء الترميز الافتراضي للنموذج يستخدم طول نص أقصر
* `--text_encoder_use_attention_mask`، لإرسال قناع الاهتمام إلى مشفر النص

### نصائح التدريب

يمكن أن يكون تدريب نموذج DeepFloyd IF أمرًا صعبًا، ولكن فيما يلي بعض النصائح التي وجدنا أنها مفيدة:

- LoRA كافٍ لتدريب نموذج المرحلة 1 لأن الدقة المنخفضة للنموذج تجعل من الصعب تمثيل التفاصيل الدقيقة على أي حال.
- بالنسبة للأشياء الشائعة أو البسيطة، فأنت لست بحاجة إلى ضبط دقة المرحلة الثانية. تأكد من تعديل المحث الذي يتم تمريره إلى أداة ضبط الدقة لإزالة الرمز الجديد من المحث الخاص بالمرحلة 1. على سبيل المثال، إذا كان محث المرحلة 1 الخاص بك هو "a sks dog" فيجب أن يكون محث المرحلة 2 الخاص بك "a dog".
- بالنسبة للتفاصيل الدقيقة مثل الوجوه، فإن ضبط دقة المرحلة 2 بالكامل أفضل من تدريب نموذج المرحلة 2 باستخدام LoRA. كما يساعد في استخدام معدلات تعلم أقل مع أحجام دفعات أكبر.
- يجب استخدام معدلات تعلم أقل لتدريب نموذج المرحلة 2.
- يعمل [DDPMScheduler] بشكل أفضل من DPMSolver المستخدم في نصوص التدريب.

## الخطوات التالية

تهانينا على تدريب نموذج DreamBooth الخاص بك! لمعرفة المزيد حول كيفية استخدام نموذجك الجديد، قد يكون الدليل التالي مفيدًا:

- تعرف على كيفية [تحميل نموذج DreamBooth](../using-diffusers/loading_adapters) للتنفيذ إذا كنت قد دربته باستخدام LoRA.