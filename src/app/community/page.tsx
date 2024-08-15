import React from "react";
import Image from "next/image";

const CommunityPage = () => {
  const communityChannels = [
    {
      name: "Discord",
      description:
        "Join our Discord server for real-time discussions and community events.",
      icon: "/images/discord.svg",
      link: "https://discord.gg/your-discord-invite",
      buttonText: "Join Discord",
    },
    {
      name: "Telegram",
      description:
        "Stay updated with our Telegram channel for quick announcements and tips.",
      icon: "/images/telegram.svg",
      link: "https://t.me/your-telegram-channel",
      buttonText: "Join Telegram",
    },
    {
      name: "WhatsApp",
      description:
        "Connect with fellow learners in our WhatsApp group for mobile-friendly discussions.",
      icon: "/images/whatsapp.svg",
      link: "https://chat.whatsapp.com/your-whatsapp-group-invite",
      buttonText: "Join WhatsApp",
    },
  ];

  return (
    <div className="container mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold text-center mb-12 dark:text-white">
        Join Our Learning Community
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {communityChannels.map(
          (channel, index) => (
            <div
              key={index}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 flex flex-col items-center"
            >
              <Image
                src={channel.icon}
                alt={`${channel.name} icon`}
                width={64}
                height={64}
                className="mb-4"
              />
              <h2 className="text-2xl font-semibold mb-2 dark:text-white">
                {channel.name}
              </h2>
              <p className="text-gray-600 dark:text-gray-300 text-center mb-6">
                {channel.description}
              </p>
              <a
                href={channel.link}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded transition duration-300"
              >
                {channel.buttonText}
              </a>
            </div>
          )
        )}
      </div>
      <div className="mt-16 text-center">
        <h2 className="text-2xl font-semibold mb-4 dark:text-white">
          Why Join Our Community?
        </h2>
        <ul className="text-left max-w-2xl mx-auto">
          <li className="mb-2 dark:text-gray-300">
            ✅ Connect with fellow learners and
            experts
          </li>
          <li className="mb-2 dark:text-gray-300">
            ✅ Get quick answers to your questions
          </li>
          <li className="mb-2 dark:text-gray-300">
            ✅ Participate in community challenges
            and events
          </li>
          <li className="mb-2 dark:text-gray-300">
            ✅ Stay updated with the latest
            learning resources
          </li>
          <li className="dark:text-gray-300">
            ✅ Share your progress and inspire
            others
          </li>
        </ul>
      </div>
    </div>
  );
};

export default CommunityPage;
