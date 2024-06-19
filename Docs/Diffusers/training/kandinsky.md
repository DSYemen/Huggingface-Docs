بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع الالتزام بالتعليمات التي قدمتها.

# كاندينسكي 2.2

يعد كاندينسكي 2.2 نموذجًا متعدد اللغات للنص إلى الصورة قادر على إنتاج صور أكثر واقعية. يتضمن النموذج نموذجًا أوليًا للصورة لإنشاء تضمينات صورة من موجهات النص، ونموذج فك تشفير يقوم بتوليد الصور بناءً على تضمينات النموذج الأولي. ولهذا السبب، ستجد نصين برمجيين منفصلين في "Diffusers" لـ "Kandinsky 2.2"، أحدهما لتدريب نموذج أولي والآخر لتدريب نموذج فك التشفير. يمكنك تدريب كلا النموذجين بشكل منفصل، ولكن للحصول على أفضل النتائج، يجب تدريب النموذجين الأولي وفك التشفير معًا.

اعتمادًا على وحدة معالجة الرسوميات (GPU) لديك، قد تحتاج إلى تمكين "gradient_checkpointing" (⚠️ غير مدعوم للنموذج الأولي!) و"mixed_precision"، و"gradient_accumulation_steps" للمساعدة في تثبيت النموذج في الذاكرة وزيادة سرعة التدريب. يمكنك تقليل استخدام الذاكرة حتى أكثر من خلال تمكين الاهتمام الفعال للذاكرة مع [xFormers] (.. / optimization / xformers) (الإصدار [v0.0.16] https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212) يفشل في التدريب على بعض وحدات معالجة الرسوميات (GPUs)، لذلك قد تحتاج إلى تثبيت إصدار التطوير بدلاً من ذلك).

يستكشف هذا الدليل النصين البرمجيين [train_text_to_image_prior.py] (https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py) و [train_text_to_image_decoder.py] (https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py) لمساعدتك على التعرف عليهما بشكل أفضل، وكيف يمكنك تكييفهما مع حالتك الاستخدام الخاصة.

قبل تشغيل النصوص البرمجية، تأكد من تثبيت المكتبة من المصدر:

بعد ذلك، انتقل إلى مجلد المثال الذي يحتوي على نص البرمجة الخاص بالتدريب وقم بتثبيت التبعيات المطلوبة للنص البرمجي الذي تستخدمه:

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة الرسوميات (GPUs) أو وحدات معالجة الرسوميات (TPUs) أو الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة والبيئة الخاصة بك. الق نظرة على جولة سريعة من 🤗 Accelerate [Quick tour] (https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

أو إذا لم تدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب] (create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص البرمجة الخاص بالتدريب.

تسلط الأقسام التالية الضوء على أجزاء من نصوص البرمجة الخاصة بالتدريب والتي تعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النصوص البرمجية بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النصوص البرمجية وإخبارنا إذا كان لديك أي أسئلة أو مخاوف.

## معلمات النص البرمجي

يوفر نص البرمجة الخاص بالتدريب العديد من المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في وظيفة ['parse_args ()'] (https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_prior.py#L190). يوفر نص البرمجة الخاص بالتدريب قيمًا افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت تريد ذلك.

على سبيل المثال، لزيادة سرعة التدريب باستخدام الدقة المختلطة بتنسيق fp16، أضف المعلمة --mixed_precision إلى أمر التدريب:

تتشابه معظم المعلمات مع المعلمات الموجودة في دليل التدريب [Text-to-image] (text2image # script-parameters)، لذلك دعنا ننتقل مباشرة إلى جولة في نصوص البرمجة الخاصة بالتدريب على كاندينسكي!

### وزن الحد الأدنى من نسبة الإشارة إلى الضوضاء

يمكن أن تساعد استراتيجية وزن الحد الأدنى من نسبة الإشارة إلى الضوضاء (Min-SNR) في التدريب عن طريق إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم نص البرمجة الخاص بالتدريب التنبؤ بـ "epsilon" (الضوضاء) أو "v_prediction"، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. تتوفر استراتيجية الترجيح هذه فقط في PyTorch وهي غير متوفرة في نص البرمجة الخاص بالتدريب على Flax.

أضف المعلمة --snr_gamma وقم بتعيينها على القيمة الموصى بها 5.0:
## نص البرنامج التدريبي:

يشبه نص البرنامج التدريبي أيضًا دليل التدريب على [النص إلى الصورة](text2image#training-script)، ولكنه تم تعديله لدعم تدريب النماذج الأولية والنماذج فك الترميز. يركز هذا الدليل على التعليمات البرمجية الفريدة الخاصة بنصوص Kandinsky 2.2 التدريبية.

يحتوي الدليل الرئيسي () على التعليمات البرمجية لإعداد مجموعة البيانات وتدريب النموذج.

أحد الاختلافات الرئيسية التي ستلاحظها على الفور هو أن نص البرنامج النصي التدريبي يحمّل أيضًا - بالإضافة إلى الجدولة والمحلل اللغوي - لمعالجة الصور قبل إدخالها في النموذج:

```py
noise_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="sample")
image_processor = CLIPImageProcessor.from_pretrained(
args.pretrained_prior_model_name_or_path, subfolder="image_processor"
)
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
args.pretrained_prior_model_name_or_path, subfolder="image_encoder", torch_dtype=weight_dtype
).eval()
text_encoder = CLIPTextModelWithProjection.from_pretrained(
args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
).eval()
```

يستخدم Kandinsky لإنشاء تضمين الصورة، لذا ستحتاج إلى إعداد المحسن لتعلم معلمات نموذج الوضع المسبق.

```بي
قبل = PriorTransformer.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")
قبل التدريب ()
المحسن = optimizer_cls (
قبل المعلمات ()،
lr = args.learning_rate،
betas = (args.adam_beta1، args.adam_beta2)،
weight_decay = args.adam_weight_decay،
eps = args.adam_epsilon،
)
```

بعد ذلك، تتم معالجة تعليقات المدخلات، وتتم معالجة الصور بواسطة:

```بي
def preprocess_train (أمثلة):
images = [image.convert ("RGB") for image in examples [image_column]]
أمثلة ["clip_pixel_values"] = image_processor (images، return_tensors = "pt"). pixel_values
أمثلة ["text_input_ids"]، أمثلة ["text_mask"] = tokenize_captions (أمثلة)
return examples
```

أخيرًا، تحول حلقة التدريب صور الإدخال إلى بيانات مضغوطة، وتضيف ضوضاء إلى تضمين الصورة، وتتنبأ بها:

```بي
model_pred = prior (
noisy_latents،
timestep = timesteps،
proj_embedding = prompt_embeds،
encoder_hidden_states = text_encoder_hidden_states،
attention_mask = text_mask،
). predicted_image_embedding
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمجدولين](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.

## نموذج فك التشفير:

يحتوي الدليل الرئيسي () على التعليمات البرمجية لإعداد مجموعة البيانات وتدريب النموذج.

على عكس نموذج الوضع المسبق، يقوم فك التشفير بتطبيق نموذج فك تشفير البيانات المضغوطة إلى صور، ويستخدم:

```بي
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
vae = VQModel.from_pretrained (
args.pretrained_decoder_model_name_or_path، subfolder = "movq"، torch_dtype = weight_dtype
). eval ()
image_encoder = CLIPVisionModelWithProjection.from_pretrained (
args.pretrained_prior_model_name_or_path، subfolder = "image_encoder"، torch_dtype = weight_dtype
). eval ()
unet = UNet2DConditionModel.from_pretrained (args.pretrained_decoder_model_name_or_path، subfolder = "unet")
```

بعد ذلك، يتضمن البرنامج النصي العديد من تحويلات الصور ووظيفة [معالجة مسبقة](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/kandinsky2_2/text_to_image/train_text_to_image_decoder.py#L622) لتطبيق التحولات على الصور وإرجاع قيم البكسل:

```بي
def preprocess_train (أمثلة):
images = [image.convert ("RGB") for image in examples [image_column]]
أمثلة ["pixel_values"] = [train_transforms (image) for image in images]
أمثلة ["clip_pixel_values"] = image_processor (images، return_tensors = "pt"). pixel_values
return examples
```

أخيرًا، تتولى حلقة التدريب تحويل الصور إلى بيانات مضغوطة، وإضافة ضوضاء، والتنبؤ ببقايا الضوضاء.

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمجدولين](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.

```بي
model_pred = unet (noisy_latents، timesteps، None، added_cond_kwargs = added_cond_kwargs). sample [:،: 4]
```

## إطلاق البرنامج النصي:

بمجرد إجراء جميع التغييرات أو كنت راضيًا عن التكوين الافتراضي، فأنت مستعد لإطلاق البرنامج النصي التدريبي! 🚀

ستقوم بالتدريب على مجموعة بيانات [تعليقات Naruto BLIP](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) لإنشاء شخصيات Naruto الخاصة بك، ولكن يمكنك أيضًا إنشاء مجموعة بيانات خاصة بك والتدريب عليها باتباع الدليل [إنشاء مجموعة بيانات للتدريب](create_dataset). قم بتعيين متغير البيئة `DATASET_NAME` إلى اسم المجموعة على Hub، أو إذا كنت تتدرب على ملفاتك الخاصة، فقم بتعيين متغير البيئة `TRAIN_DIR` إلى مسار لمجموعة البيانات الخاصة بك.

إذا كنت تتدرب على أكثر من وحدة معالجة مركزية (GPU)، فأضف المعلمة --multi_gpu إلى الأمر accelerate launch.

<Tip>

لمراقبة تقدم التدريب باستخدام Weights & Biases، أضف المعلمة --report_to=wandb إلى أمر التدريب. ستحتاج أيضًا إلى إضافة --validation_prompt إلى أمر التدريب لتتبع النتائج. يمكن أن يكون هذا مفيدًا جدًا في تصحيح أخطاء النموذج وعرض النتائج الوسيطة.

</Tip>

بمجرد الانتهاء من التدريب، يمكنك استخدام نموذجك المدرب حديثًا للاستنتاج!

## نموذج الوضع المسبق:

```بي
from diffusers import AutoPipelineForText2Image، DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained (output_dir، torch_dtype = torch.float16)
prior_components = {"prior_" + k: v for k، v في prior_pipeline.components.items ()}
pipeline = AutoPipelineForText2Image.from_pretrained ("kandinsky-community/kandinsky-2-2-decoder"، ** prior_components، torch_dtype = torch.float16)

pipe.enable_model_cpu_offload ()
prompt = "A robot naruto، 4k photo"
image = pipeline (prompt = prompt، negative_prompt = negative_prompt). images [0]
```

<Tip>

لا تتردد في استبدال "kandinsky-community/kandinsky-2-2-decoder" بنقطة تفتيش فك التشفير المدربة الخاصة بك!

</Tip>

## نموذج فك التشفير:

```بي
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained ("path/to/saved/model"، torch_dtype = torch.float16)
pipeline.enable_model_cpu_offload ()

prompt = "A robot naruto، 4k photo"
image = pipeline (prompt = prompt). images [0]
```

بالنسبة لنموذج فك التشفير، يمكنك أيضًا إجراء الاستدلال من نقطة تفتيش محفوظة، والتي يمكن أن تكون مفيدة لعرض النتائج الوسيطة. في هذه الحالة، قم بتحميل نقطة التفتيش في UNet:

```بي
from diffusers import AutoPipelineForText2Image، UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained ("path/to/saved/model" + "/checkpoint-<N>/unet")

pipeline = AutoPipelineForText2Image.from_pretrained ("kandinsky-community/kandinsky-2-2-decoder"، unet = unet، torch_dtype = torch.float16)
pipeline.enable_model_cpu_offload ()

image = pipeline (prompt = "A robot naruto، 4k photo"). images [0]
```

## الخطوات التالية:

تهانينا على تدريب نموذج Kandinsky 2.2! لمعرفة المزيد حول كيفية استخدام نموذجك الجديد، قد تكون الأدلة التالية مفيدة:

- اقرأ الدليل [Kandinsky](../using-diffusers/kandinsky) لمعرفة كيفية استخدامه لمجموعة متنوعة من المهام المختلفة (النص إلى الصورة، الصورة إلى الصورة، الإكمال، الاستيفاء)، وكيف يمكن دمجه مع ControlNet.
- تحقق من أدلة التدريب [DreamBooth](dreambooth) و [LoRA](lora) لمعرفة كيفية تدريب نموذج Kandinsky شخصي باستخدام عدد قليل فقط من الصور. يمكن حتى دمج تقنيتي التدريب هاتين!