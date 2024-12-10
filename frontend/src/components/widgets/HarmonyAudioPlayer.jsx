import React from 'react';
import AudioPlayer from 'react-h5-audio-player';
import 'react-h5-audio-player/lib/styles.css';
import './HarmonyAudioPlayer.css'

const HarmonyAudioPlayer = ({ src }) => {
    return (
        <div className="w-full">
            <AudioPlayer
                src={src}
                className="bg-neutral-700 text-orange-400 rounded-lg p-2"
                // Custom styling via CSS or Tailwind
                style={{
                    // Example: Override some default styles
                    borderRadius: '0',
                }}
                // Disable default styles if needed
                layout="horizontal"
                showJumpControls={false}
                customAdditionalControls={[]}
                customVolumeControls={[]}
                // Add more customization as needed
            />
        </div>
    );
};

export default HarmonyAudioPlayer;