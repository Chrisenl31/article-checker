from django.db import models
from django.contrib.auth.models import User
import uuid


class PasswordReset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    reset_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    created_when = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Password reset for {self.user.username} at {self.created_when}"

class Titles(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="titles")
    title = models.CharField(max_length=255)

    def __str__(self):
        return self.title

class Abstracts(models.Model):
    title = models.OneToOneField(Titles, on_delete=models.CASCADE, related_name="abstracts")
    abstract = models.TextField()

    def __str__(self):
        return f"Abstract of {self.title}"
    
class Keywords(models.Model):
    abstract = models.OneToOneField(Abstracts, on_delete=models.CASCADE, related_name="keywords")
    keyword_list = models.CharField(max_length=255)

    def __str__(self):
        return f"Keywords for {self.abstract.title}"