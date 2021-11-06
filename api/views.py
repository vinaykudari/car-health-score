from PIL import Image
from django.http import JsonResponse
from rest_framework.views import APIView

from api.helper import analyse


class API(APIView):
    def get(self, request):
        response = {
            "error": "Method GET does not exists"
        }

        return JsonResponse(response, status=400)

    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return JsonResponse(
                {
                    "error": "Missing file"
                }, status=400
            )

        try:
            Image.open(image)
        except Exception as e:
            return JsonResponse(
                {
                    "error": "File provided is either corrupt or not a valid Image"
                }, status=400
            )

        damage_detected, image_details = analyse(image)

        response = {
            'damage_detected': damage_detected,
            'image_details': image_details,
        }

        return JsonResponse(response)
