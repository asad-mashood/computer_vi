# Eight-Method Comparison Results

| Method           | Noise Handling    | Edge Quality              | Boundary Detection      | Overall Performance         |
|:-----------------|:------------------|:--------------------------|:------------------------|:----------------------------|
| Original + Sobel | F1 stability 0.34 | 5.2% edge pixels; inspect | 5/5 candidates; inspect | Needs visual review         |
| Original + Canny | F1 stability 0.36 | 3.0% edge pixels; inspect | 5/5 candidates; inspect | Needs visual review         |
| Average + Sobel  | F1 stability 0.85 | 1.0% edge pixels; inspect | 3/5 candidates; inspect | Needs visual review         |
| Average + Canny  | F1 stability 0.79 | 0.3% edge pixels; inspect | 2/5 candidates; inspect | Needs visual review         |
| Gaussian + Sobel | F1 stability 0.86 | 2.3% edge pixels; inspect | 5/5 candidates; inspect | Promising — review outlines |
| Gaussian + Canny | F1 stability 0.74 | 1.2% edge pixels; inspect | 2/5 candidates; inspect | Needs visual review         |
| Median + Sobel   | F1 stability 0.77 | 1.6% edge pixels; inspect | 4/5 candidates; inspect | Promising — review outlines |
| Median + Canny   | F1 stability 0.62 | 0.9% edge pixels; inspect | 3/5 candidates; inspect | Needs visual review         |