بالتأكيد، سأقوم بترجمة النص المطلوب مع اتباع التعليمات المذكورة.

# تدريب نموذج الانتشار

تعد التوليد غير المشروط للصور تطبيقًا شائعًا لنماذج الانتشار التي تولد صورًا تشبه تلك الموجودة في مجموعة البيانات المستخدمة للتدريب. وعادة ما يتم الحصول على أفضل النتائج من خلال الضبط الدقيق لنموذج مُدرب مسبقًا على مجموعة بيانات محددة. يمكنك العثور على العديد من هذه النقاط المرجعية على [Hub] (https://huggingface.co/search/full-text؟q=unconditional-image-generation&type=model)، ولكن إذا لم تتمكن من العثور على واحدة تناسبك، فيمكنك دائمًا تدريب النموذج الخاص بك!

سيوضح هذا البرنامج التعليمي كيفية تدريب [`UNet2DModel`] من الصفر على جزء من مجموعة بيانات [Smithsonian Butterflies] (https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) لتوليد الفراشات الخاصة بك 🦋.

<Tip>

💡 يعتمد هذا البرنامج التعليمي للتدريب على دفتر الملاحظات [التدريب مع 🧨 Diffusers] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb). للحصول على تفاصيل وسياق إضافي حول نماذج الانتشار مثل كيفية عملها، راجع دفتر الملاحظات!

</Tip>

قبل البدء، تأكد من تثبيت 🤗 Datasets لتحميل مجموعات بيانات الصور ومعالجتها مسبقًا، و🤗 Accelerate، لتبسيط التدريب على أي عدد من وحدات معالجة الرسومات (GPU). سيقوم الأمر التالي أيضًا بتثبيت [TensorBoard] (https://www.tensorflow.org/tensorboard) لتصور مقاييس التدريب (يمكنك أيضًا استخدام [Weights & Biases] (https://docs.wandb.ai/) لتتبع التدريب الخاص بك).

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install diffusers [training]
```

نحن نشجعك على مشاركة نموذجك مع المجتمع، وللقيام بذلك، ستحتاج إلى تسجيل الدخول إلى حساب Hugging Face الخاص بك (قم بإنشاء واحد [هنا] (https://hf.co/join) إذا لم يكن لديك واحد بالفعل!). يمكنك تسجيل الدخول من دفتر الملاحظات وإدخال رمزك عند المطالبة. تأكد من أن لديك دور الكتابة.

```py
>>> من huggingface_hub import notebook_login

>>> notebook_login()
```

أو تسجيل الدخول من المحطة الطرفية:

```bash
huggingface-cli login
```

نظرًا لأن نقاط التحقق من النموذج كبيرة جدًا، فقم بتثبيت [Git-LFS] (https://git-lfs.com/) لإصدار هذه الملفات الكبيرة:

```bash
! sudo apt -qq install git-lfs
! git config --global credential.helper store
```

## تكوين التدريب

للراحة، قم بإنشاء فئة `TrainingConfig` تحتوي على فرط معلمات التدريب (يمكنك ضبطها حسب رغبتك):

```py
>>> من dataclasses import dataclass

>>> @ dataclass
... class TrainingConfig:
... image_size = 128 # دقة الصورة المولدة
... train_batch_size = 16
... eval_batch_size = 16 # عدد الصور التي سيتم أخذ عينات منها أثناء التقييم
... num_epochs = 50
... gradient_accumulation_steps = 1
... learning_rate = 1e-4
... lr_warmup_steps = 500
... save_image_epochs = 10
... save_model_epochs = 30
... mixed_precision = "fp16" # `no` لـ float32، `fp16` لـ automatic mixed precision
... output_dir = "ddpm-butterflies-128" # اسم النموذج محليًا وعلى HF Hub

... push_to_hub = True # ما إذا كان سيتم تحميل النموذج المحفوظ إلى HF Hub
... hub_model_id = "<your-username>/<my-awesome-model>" # اسم المستودع الذي سيتم إنشاؤه على HF Hub
... hub_private_repo = False
... overwrite_output_dir = True # الكتابة فوق النموذج القديم عند إعادة تشغيل دفتر الملاحظات
... seed = 0


>>> config = TrainingConfig()
```

## تحميل مجموعة البيانات

يمكنك تحميل مجموعة بيانات [Smithsonian Butterflies] (https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) بسهولة باستخدام مكتبة 🤗 Datasets:

```py
>>> من datasets import load_dataset

>>> config.dataset_name = "huggan/smithsonian_butterflies_subset"
>>> dataset = load_dataset (config.dataset_name، split = "train")
```

<Tip>

💡 يمكنك العثور على مجموعات بيانات إضافية من [حدث HugGan Community] (https://huggingface.co/huggan) أو يمكنك استخدام مجموعة البيانات الخاصة بك عن طريق إنشاء [`ImageFolder`] محلي (https://huggingface.co/docs/datasets/image_dataset#imagefolder). قم بتعيين `config.dataset_name` إلى معرف المستودع لمجموعة البيانات إذا كانت من حدث HugGan Community، أو `imagefolder` إذا كنت تستخدم صورك الخاصة.

</Tip>

يستخدم 🤗 Datasets ميزة [`~ datasets.Image`] لفك ترميز بيانات الصورة وتحميلها كـ [`PIL.Image`] (https://pillow.readthedocs.io/en/stable/reference/Image.html) والتي يمكننا تصورها:

```py
>>> import matplotlib.pyplot as plt

>>> fig، axs = plt.subplots (1، 4، figsize = (16، 4))
>>> for i، image in enumerate (dataset [: 4] ["image"]):
... axs [i].imshow (image)
... axs [i].set_axis_off ()
>>> fig.show ()
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_ds.png"/>
</div>

ومع ذلك، فإن الصور بأحجام مختلفة، لذلك ستحتاج إلى معالجتها مسبقًا:

* `Resize` يغير حجم الصورة إلى الحجم المحدد في `config.image_size`.
* `RandomHorizontalFlip` يزيد من حجم مجموعة البيانات عن طريق عكس الصور بشكل عشوائي.
* `Normalize` مهم لإعادة تحجيم قيم البكسل إلى نطاق [-1، 1]، وهو ما يتوقعه النموذج.

```py
>>> من torchvision import transforms

>>> preprocess = transforms.Compose
... (
... [
... transforms.Resize ((config.image_size، config.image_size))،
... transforms.RandomHorizontalFlip ()،
... transforms.ToTensor ()،
... transforms.Normalize ([0.5]، [0.5])،
... ]
... )
```

استخدم طريقة [`~ datasets.Dataset.set_transform`] من 🤗 Datasets لتطبيق وظيفة `preprocess` أثناء التنقل أثناء التدريب:

```py
>>> def transform (examples):
... images = [preprocess (image.convert ("RGB")) for image in examples ["image"]]
... return {"images": images}


>>> dataset.set_transform (transform)
```

لا تتردد في تصور الصور مرة أخرى للتأكد من تغيير حجمها. الآن أنت مستعد لتغليف مجموعة البيانات في [DataLoader] (https://pytorch.org/docs/stable/data#torch.utils.data.DataLoader) للتدريب!

```py
>>> import torch

>>> train_dataloader = torch.utils.data.DataLoader (dataset، batch_size = config.train_batch_size، shuffle = True)
```

## إنشاء UNet2DModel

من السهل إنشاء نماذج مُدربة مسبقًا في 🧨 Diffusers من فئة النموذج الخاصة بها باستخدام المعلمات التي تريدها. على سبيل المثال، لإنشاء [`UNet2DModel`]:

```py
>>> من diffusers import UNet2DModel

>>> model = UNet2DModel (
... sample_size = config.image_size، # دقة الصورة المستهدفة
... in_channels = 3، # عدد قنوات الإدخال، 3 للصور RGB
... out_channels = 3، # عدد قنوات الإخراج
... layers_per_block = 2، # عدد طبقات ResNet المستخدمة لكل كتلة UNet
... block_out_channels = (128، 128، 256، 256، 512، 512)، # عدد قنوات الإخراج لكل كتلة UNet
... down_block_types = (
... "DownBlock2D"، # كتلة انخفاض ResNet العادية
... "DownBlock2D"،
... "DownBlock2D"،
... "DownBlock2D"،
... "AttnDownBlock2D"، # كتلة انخفاض ResNet مع الانتباه المكاني الذاتي
... "DownBlock2D"،
... )،
... up_block_types = (
... "UpBlock2D"، # كتلة ResNet عادية
... "AttnUpBlock2D"، # كتلة ResNet مع الانتباه المكاني الذاتي
... "UpBlock2D"،
... "UpBlock2D"،
... "UpBlock2D"،
... "UpBlock2D"،
... )،
... )
```

غالبًا ما يكون من الجيد التحقق بسرعة من أن شكل صورة الإدخال يتطابق مع شكل إخراج النموذج:

```py
>>> sample_image = dataset [0] ["images"].unsqueeze (0)
>>> print ("شكل الإدخال:"، sample_image.shape)
شكل الإدخال: torch.Size ([1، 3، 128، 128])

>>> print ("شكل الإخراج:"، model (sample_image، timestep = 0). sample.shape)
شكل الإخراج: torch.Size ([1، 3، 128، 128])
```

رائع! بعد ذلك، ستحتاج إلى جدول زمني لإضافة بعض الضوضاء إلى الصورة.

## إنشاء جدول زمني

يتصرف الجدول الزمني بشكل مختلف اعتمادًا على ما إذا كنت تستخدم النموذج للتدريب أو الاستدلال. أثناء الاستدلال، يقوم الجدول الزمني بتوليد الصورة من الضوضاء. أثناء التدريب، يأخذ الجدول الزمني إخراج النموذج - أو عينة - من نقطة محددة في عملية الانتشار ويضيف ضوضاء إلى الصورة وفقًا لـ *جدول زمني للضوضاء* و *قاعدة تحديث*.

دعونا نلقي نظرة على [`DDPMScheduler`] واستخدام طريقة `add_noise` لإضافة بعض الضوضاء العشوائية إلى `sample_image` من قبل:

```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import DDPMScheduler

>>> noise_scheduler = DDPMScheduler (num_train_timesteps = 1000)
>>> noise = torch.randn (sample_image.shape)
>>> timesteps = torch.LongTensor ([50])
>>> noisy_image = noise_scheduler.add_noise (sample_image، noise، timesteps)

>>> Image.fromarray (((noisy_image.permute (0، 2، 3، 1) + 1.0) * 127.5). type (torch.uint8). numpy () [0])
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/noisy_butterfly.png"/>
</div>

الهدف من التدريب على النموذج هو التنبؤ بالضوضاء المضافة إلى الصورة. يمكن حساب الخسارة في هذه الخطوة على النحو التالي:

```py
>>> import torch.nn.functional as F

>>> noise_pred = model (noisy_image، timesteps). sample
>>> loss = F.mse_loss (noise_pred، noise)
```
## تدريب النموذج

الآن، لديك معظم القطع اللازمة لبدء تدريب النموذج وكل ما تبقى هو جمع كل شيء معًا.

أولاً، ستحتاج إلى محسن ومخطط لمعدل التعلم:

ثم، ستحتاج إلى طريقة لتقييم النموذج. للتقييم، يمكنك استخدام [`DDPMPipeline`] لتوليد دفعة من الصور النموذجية وحفظها كشبكة:

الآن يمكنك لف كل هذه المكونات معًا في حلقة تدريب باستخدام 🤗 Accelerate لتسجيل TensorBoard سهل، وتراكم التدرجات، والتدريب على الدقة المختلطة. لتحميل النموذج إلى Hub، قم بكتابة دالة للحصول على اسم مستودعك ومعلوماتك، ثم قم بالدفع إلى Hub.

<Tip>
💡 قد تبدو حلقة التدريب أدناه مخيفة وطويلة، ولكنها ستكون جديرة بالاهتمام لاحقًا عندما تطلق التدريب في سطر واحد فقط من التعليمات البرمجية! إذا لم تتمكن من الانتظار وكنت تريد البدء في إنشاء الصور، فلا تتردد في نسخ ولصق وتشغيل التعليمات البرمجية أدناه. يمكنك دائمًا العودة وفحص حلقة التدريب عن كثب لاحقًا، مثل عندما تنتظر نموذجك لإنهاء التدريب. 🤗
</Tip>

Phew، كان هذا الكثير من التعليمات البرمجية! ولكنك أخيرًا مستعد لإطلاق التدريب باستخدام وظيفة [`~accelerate.notebook_launcher`] في 🤗 Accelerate. قم بتمرير الدالة حلقة التدريب، وجميع الحجج التدريب، وعدد العمليات (يمكنك تغيير هذه القيمة إلى عدد وحدات معالجة الرسومات المتوفرة لديك) لاستخدامها في التدريب:

بمجرد اكتمال التدريب، الق نظرة على الصور النهائية 🦋 التي تم إنشاؤها بواسطة نموذج الانتشار الخاص بك!

## الخطوات التالية

يعد إنشاء الصور غير المشروط مثالًا واحدًا على المهمة التي يمكن تدريبها. يمكنك استكشاف مهام وتقنيات تدريب أخرى من خلال زيارة صفحة [🧨 أمثلة التدريب على Diffusers](../training/overview). فيما يلي بعض الأمثلة على ما يمكنك تعلمه:

* [العكس النصي](../training/text_inversion)، وهو خوارزمية تعلم النموذج مفهومًا بصريًا محددًا وتكامله في الصورة المولدة.
* [DreamBooth](../training/dreambooth)، وهي تقنية لتوليد صور شخصية لموضوع معين بناءً على عدة صور إدخال للموضوع.
* [دليل](../training/text2image) إلى ضبط نموذج الانتشار المستقر على مجموعة البيانات الخاصة بك.
* [دليل](../training/lora) لاستخدام LoRA، وهي تقنية فعالة من حيث الذاكرة لضبط النماذج الكبيرة جدًا بشكل أسرع.