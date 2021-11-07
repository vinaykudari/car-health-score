from PIL import Image
from django.http import JsonResponse
from rest_framework.views import APIView
import cv2
import numpy as np

from api.helper import analyse


class API(APIView):
    def get(self, request):
        response = {
            "error": "Method GET does not exists"
        }

        return JsonResponse(response, status=400)

    def post(self, request):
        image = request.FILES.get('image')
        use_max_area = request.POST.get('use_max_area')

        if not image:
            return JsonResponse(
                {
                    "error": "Missing file"
                }, status=400
            )

        try:
            image = cv2.imdecode(
                np.fromstring(image.read(), np.uint8),
                cv2.IMREAD_UNCHANGED,
            )

            scale_percent = 50
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        except Exception as e:
            print(e)
            return JsonResponse(
                {
                    "error": "File provided is either corrupt or not a valid Image"
                }, status=400
            )

        damage_detected, image_details = analyse(image, use_max_area)

        response = {
            'damage_detected': damage_detected,
            'image_details': image_details,
        }

        return JsonResponse(response)
