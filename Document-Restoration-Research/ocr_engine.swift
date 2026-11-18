
import Foundation
import Vision
import ImageIO

guard CommandLine.arguments.count >= 2 else {
    print("Usage: ocr_engine <image_path>")
    exit(1)
}

let imagePath = CommandLine.arguments[1]
let imageURL = URL(fileURLWithPath: imagePath)

guard let imageSource = CGImageSourceCreateWithURL(
    imageURL as CFURL, nil
),
let cgImage = CGImageSourceCreateImageAtIndex(
    imageSource, 0, nil
) else {
    print("ERROR: Could not load image")
    exit(1)
}

let request = VNRecognizeTextRequest()

request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US"]

do {
    let handler = VNImageRequestHandler(
        cgImage: cgImage,
        options: [:]
    )

    try handler.perform([request])

    let observations = request.results ?? []

    for observation in observations {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }

} catch {
    print("OCR ERROR: \(error)")
    exit(1)
}
