'use client';
import { Button } from '@/components/ui/button';
import { useState, useEffect, useRef } from 'react';

export default function HeroWithCanvasReveal() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRevealing, setIsRevealing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const image = new Image();
    image.src =
      'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Screenshot%202025-02-15%20at%2012.48.18%E2%80%AFAM-446vARkwukyrUdBkiSPetwpvisaJHR.png';
    image.crossOrigin = 'anonymous';

    image.onload = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;

      let progress = 0;
      const revealSpeed = 0.003;

      function animate() {
        if (!ctx || !isRevealing) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the full image
        ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvas.width, canvas.height);

        // Create a radial gradient for the mask
        const gradient = ctx.createRadialGradient(
          canvas.width / 2,
          canvas.height / 2,
          0,
          canvas.width / 2,
          canvas.height / 2,
          Math.max(canvas.width, canvas.height) * progress
        );
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        gradient.addColorStop(0.8, 'rgba(0, 0, 0, 0.7)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 1)');

        // Apply the mask
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw single radar pulse effect
        function drawRadarEffect(ctx: CanvasRenderingContext2D, progress: number) {
          const centerX = canvas.width / 2;
          const centerY = canvas.height / 2;
          const maxRadius = Math.max(canvas.width, canvas.height) / 2;

          // Single pulse that moves faster than the reveal
          const radarSpeed = 1.75;
          const pulseProgress = progress * radarSpeed;

          if (pulseProgress <= 1) {
            const currentRadius = maxRadius * pulseProgress;
            // Fade out as the pulse expands
            const opacity = Math.max(0, 0.8 - pulseProgress);

            // Main circle
            ctx.beginPath();
            ctx.arc(centerX, centerY, currentRadius, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(0, 255, 100, ${opacity})`;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Inner circle
            ctx.beginPath();
            ctx.arc(centerX, centerY, currentRadius * 0.95, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(0, 255, 100, ${opacity * 0.6})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
        drawRadarEffect(ctx, progress);

        progress += revealSpeed;

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      }

      if (isRevealing) {
        animate();
      }
    };

    const handleResize = () => {
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [isRevealing]);

  const handleReveal = () => {
    setIsRevealing(true);
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-black">
      {/* Canvas for image reveal effect */}
      <canvas ref={canvasRef} className="absolute inset-0 z-0" />

      {/* Gradient Overlay */}
      <div className="absolute inset-0 z-10 bg-gradient-to-b from-black/30 via-transparent to-black/30" />

      {/* Hero Content */}
      <div className="relative z-20 text-center px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl sm:text-5xl lg:text-7xl font-extrabold text-white tracking-tight mb-4">Theia</h1>
        <p className="mt-6 text-xl sm:text-2xl text-gray-300 max-w-3xl mx-auto">
          Experience the power of X-ray vision reimagined.
        </p>
        <div className="mt-10">
          <Button
            size="lg"
            onClick={handleReveal}
            className="bg-cyan-500 hover:bg-cyan-400 text-white font-semibold py-3 px-8 rounded-lg text-lg transition duration-300 ease-in-out transform hover:scale-105"
          >
            {isRevealing ? 'Get Started' : 'Reveal!'}
          </Button>
        </div>
      </div>
    </div>
  );
}
