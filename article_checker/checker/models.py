import uuid

from django.contrib.auth.models import User
from django.db import models


class PasswordReset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    reset_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    created_when = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Password reset for {self.user.username} at {self.created_when}"

class AbstractInput(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    inputID = models.AutoField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.TextField(max_length=255)
    abstract=models.TextField(max_length=450)
    keywords=models.TextField(max_length=255)
    result=models.TextField(max_length=450)

    def __str__(self):
        return f"Abstract saved"