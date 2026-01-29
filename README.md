## qqes notes

Le `pixel_array` des datasets peut présenter une dimension différente dans la direction z (sens tete-pied soit axe=0 pour array numpy) 
que celle réellement prise par le volume PET.
  ``` python
  assert ds.pixel_array.shape, (654, 256, 256)
  assert len(ds.PerFrameFunctionnalGroupSequence), 327
  ```
Cela signifie que le `pixel_array` represente *deux* volumes dont la taille est [327, 256, 256].

   
