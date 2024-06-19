# التدريب الموزع باستخدام 🤗 Accelerate

مع تزايد حجم النماذج، برزت الموازاة كاستراتيجية لتدريب النماذج الأكبر على أجهزة محدودة، وتسريع سرعة التدريب بعدة رتب من الحجم. وفي Hugging Face، قمنا بإنشاء مكتبة [🤗 Accelerate](https://huggingface.co/docs/accelerate) لمساعدة المستخدمين على تدريب نموذج 🤗 Transformers بسهولة على أي نوع من الإعدادات الموزعة، سواء كانت عدة وحدات معالجة رسومية (GPU) على جهاز واحد أو عدة وحدات معالجة رسومية عبر عدة أجهزة. في هذا الدرس، تعلم كيفية تخصيص حلقة التدريب الأصلية لـ PyTorch لتمكين التدريب في بيئة موزعة.

## الإعداد

ابدأ بتثبيت 🤗 Accelerate:

```bash
pip install accelerate
```

ثم قم باستيراد وإنشاء كائن [`~accelerate.Accelerator`]. سيقوم كائن [`~accelerate.Accelerator`] تلقائيًا باكتشاف نوع إعدادك الموزع وتحسين جميع المكونات اللازمة للتدريب. لا تحتاج إلى وضع نموذجك على جهاز بشكل صريح.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## الاستعداد للتسريع

الخطوة التالية هي تمرير جميع كائنات التدريب ذات الصلة إلى طريقة [`~accelerate.Accelerator.prepare`]. ويشمل ذلك DataLoaders للتدريب والتقييم، ونموذجًا، ومُحَسِّنًا:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## الخلفي

الإضافة الأخيرة هي استبدال "loss.backward()" النموذجية في حلقة التدريب الخاصة بك بطريقة 🤗 Accelerate [`~accelerate.Accelerator.backward`]:

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

كما ترون في التعليمات البرمجية التالية، كل ما تحتاجه هو إضافة أربع أسطر من التعليمات البرمجية إلى حلقة التدريب الخاصة بك لتمكين التدريب الموزع!

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## التدريب

بمجرد إضافة أسطر التعليمات البرمجية ذات الصلة، قم بتشغيل تدريبك في نص برمجي أو دفتر ملاحظات مثل Colaboratory.

### التدريب باستخدام نص برمجي

إذا كنت تقوم بتشغيل تدريبك من نص برمجي، فقم بتشغيل الأمر التالي لإنشاء وحفظ ملف تكوين:

```bash
accelerate config
```

ثم قم بتشغيل تدريبك باستخدام:

```bash
accelerate launch train.py
```

### التدريب باستخدام دفتر ملاحظات

يمكن لـ 🤗 Accelerate أيضًا تشغيله في دفتر ملاحظات إذا كنت تخطط لاستخدام وحدات معالجة الرسومات (TPUs) الخاصة بـ Colaboratory. قم بتغليف جميع التعليمات البرمجية المسؤولة عن التدريب في دالة، ومررها إلى [`~accelerate.notebook_launcher`]:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

لمزيد من المعلومات حول 🤗 Accelerate وميزاته الغنية، يرجى زيارة [الوثائق](https://huggingface.co/docs/accelerate).