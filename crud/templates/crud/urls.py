from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predict.urls')),  # This makes mallapp's URLs available at the root
]